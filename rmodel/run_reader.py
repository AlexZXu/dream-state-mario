from rmodel.reader import Reader, ReaderProps, reader_loss
from rmodel.train_reader import precision_recall
from envs.smb_events import event_labels, MAX_DX
from envs.smb_sequences import FRAME_SKIP, CONTEXT_FRAMES, X_POS_SCALE, level_id
from envs.smb import load
from envs.mario import decode, DYING, DEAD
from play.harness import Harness, level_starts, START_LIVES, DIED_STEPS, CLEAR_STEPS

import numpy as np
import torch


def check_reader():
    props = ReaderProps()
    reader = Reader(props)

    num_params = sum(param.numel() for param in reader.parameters())
    print(f"reader params {num_params / 1e6:.2f}M")
    assert num_params < 1_000_000

    pair = torch.randn(4, 2, props.latent_channels, 28, 32)
    output = reader(pair)

    assert output["dx"].shape == (4,)
    assert output["died"].shape == (4,)
    assert output["clear"].shape == (4,)
    assert output["powerup"].shape == (4, props.num_powerups)

    batch = {
        "dx": torch.tensor([0.5, -0.5, 0.0, 1.0]),
        "dx_valid": torch.tensor([1.0, 1.0, 1.0, 0.0]),
        "died": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "clear": torch.tensor([0.0, 1.0, 0.0, 0.0]),
        "powerup": torch.tensor([0, 1, 2, 0])
    }

    # A perfect answer costs nothing, and the masked dx is free to be anything.
    perfect = {
        "dx": torch.tensor([0.5, -0.5, 0.0, -7.0]),
        "died": 100 * (2 * batch["died"] - 1),
        "clear": 100 * (2 * batch["clear"] - 1),
        "powerup": 100 * torch.nn.functional.one_hot(batch["powerup"], 3).float()
    }
    assert reader_loss(perfect, batch).item() < 1e-4

    wrong = dict(perfect)
    wrong["dx"] = torch.tensor([0.6, -0.5, 0.0, -7.0])
    assert abs(reader_loss(wrong, batch).item() - 100 * 0.1 ** 2 / 3) < 1e-4

    loss = reader_loss(output, batch)
    loss.backward()

    missing = [name for name, param in reader.named_parameters() if (param.grad is None)]
    assert missing == [], f"no gradient reached {missing}"

    # 3 predicted, 2 of them right, out of 4 real positives
    logits = torch.tensor([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
    target = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    precision, recall = precision_recall(logits, target)
    assert abs(precision - 2 / 3) < 1e-6 and abs(recall - 0.5) < 1e-6

    print("reader passed")


def check_labels():
    data = load()
    episodes = data["episodes"]
    ram = np.asarray(data["ram"])

    labels = event_labels(ram, episodes)
    state = decode(ram)
    end = episodes["start"] + episodes["length"]

    # Every failed run has to be labelled dead at its end, pits included, and no run
    # may be labelled dead in its first second.
    for episode, last in zip(episodes, end - 1):
        if (not episode["win"]):
            assert labels["died"][last]

        assert not labels["died"][episode["start"]:episode["start"] + 60].any()

    pits = [
        episode for episode in episodes[~episodes["win"]]
        if (not np.any(state["player_state"][episode["start"]:episode["start"] + episode["length"]] == DYING))
    ]
    pit_frames = [labels["died"][e["start"]:e["start"] + e["length"]].sum() for e in pits]
    print(f"{len(pits)} pit deaths, labelled dead for {min(pit_frames)} to {max(pit_frames)} frames each")

    # The game spends about 4 seconds on a pit death, and the harness needs half a
    # second of it to respawn.
    assert min(pit_frames) >= 200 and max(pit_frames) <= 500

    # Every flagpole win ends labelled clear; a failed run is never clear.
    flag_wins = episodes[episodes["win"] & (episodes["stage"] != 3)]
    flagged = sum(labels["clear"][e["start"] + e["length"] - 1] for e in flag_wins)
    print(f"{flagged}/{len(flag_wins)} flagpole wins end labelled clear")
    assert flagged >= len(flag_wins) - 4

    for episode in episodes[~episodes["win"]]:
        assert not labels["clear"][episode["start"]:episode["start"] + episode["length"]].any()

    assert not np.any(labels["died"] & labels["clear"])

    # dx is the x_pos difference one step back, wherever it is marked valid.
    t = np.flatnonzero(labels["dx_valid"])[50_000]
    assert labels["dx"][t] * MAX_DX == state["x_pos"][t] - state["x_pos"][t - FRAME_SKIP]
    assert np.abs(labels["dx"]).max() <= 1
    assert labels["dx_valid"].mean() > 0.99

    for name in ("died", "clear"):
        print(f"{name}: {labels[name].mean() * 100:.1f}% of frames")

    print("labels passed")

    return episodes


def check_harness(episodes):
    starts = level_starts(episodes)
    span = FRAME_SKIP * CONTEXT_FRAMES + FRAME_SKIP - 1

    assert sorted(starts.keys()) == list(range(32))

    for level, t in starts.items():
        e = np.searchsorted(episodes["start"], t, side="right") - 1
        assert level_id(episodes["world"][e], episodes["stage"][e]) == level
        assert t == episodes["start"][e] + span

    harness = Harness()

    # Walking: x_pos integrates and nothing fires.
    for _ in range(100):
        assert harness.update(dx=4.0, died=0.1, clear=0.0, powerup=1) is None

    assert harness.x_pos == 400.0
    assert harness.conditioning() == {"level": 0, "x_pos": 400.0 / X_POS_SCALE, "powerup": 1}

    # A flicker of "died" shorter than the debounce does nothing.
    for _ in range(DIED_STEPS - 1):
        assert harness.update(0.0, 0.9, 0.0, 1) is None

    assert harness.update(0.0, 0.1, 0.0, 1) is None
    assert harness.lives == START_LIVES

    # A real death: respawn on the DIED_STEPS-th step, one life gone, position reset.
    for _ in range(DIED_STEPS - 1):
        assert harness.update(0.0, 0.9, 0.0, 1) is None

    assert harness.update(0.0, 0.9, 0.0, 1) == "respawn"
    assert (harness.lives, harness.x_pos, harness.powerup, harness.level) == (START_LIVES - 1, 0.0, 0, 0)

    # Clearing a level moves on and keeps the lives.
    events = [harness.update(2.0, 0.0, 0.9, 0) for _ in range(CLEAR_STEPS)]
    assert events[:-1] == [None] * (CLEAR_STEPS - 1) and events[-1] == "next_level"
    assert (harness.level, harness.lives, harness.x_pos) == (1, START_LIVES - 1, 0.0)

    # Losing the remaining lives ends the game and starts over from 1-1.
    events = [harness.update(0.0, 0.9, 0.0, 0) for _ in range(DIED_STEPS * (START_LIVES - 1))]
    assert [event for event in events if (event is not None)] == ["respawn"] * (START_LIVES - 2) + ["game_over"]
    assert (harness.level, harness.lives) == (0, START_LIVES)

    # 8-4 wraps to 1-1, and x_pos never goes negative.
    last = Harness(level=31)
    assert [last.update(-5.0, 0.0, 0.9, 0) for _ in range(CLEAR_STEPS)][-1] == "next_level"
    assert last.level == 0

    assert Harness().update(-5.0, 0.0, 0.0, 0) is None

    print("harness passed")


def main():
    torch.manual_seed(0)

    check_reader()
    episodes = check_labels()
    check_harness(episodes)

    print("\nAll tests passed.")

if (__name__ == "__main__"):
    main()
