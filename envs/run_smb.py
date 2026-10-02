from envs.mario import decode, BUTTONS, CROP_TOP, FRAME_H, FRAME_W, DEAD
from envs.smb import load, scan, read_ram, to_action, NAME, NUM_COLORS, RAM_SIZE, FPS
import os

import numpy as np
from PIL import Image

NUM_SAMPLES = 300


def main():
    data = load()

    frames = data["frames"]
    actions = data["actions"]
    ram = data["ram"]
    episodes = data["episodes"]
    palette = data["palette"]

    N = frames.shape[0]
    print(f"frames {frames.shape}  actions {actions.shape}  ram {ram.shape}  episodes {episodes.shape}")

    assert frames.shape == (N, FRAME_H, FRAME_W) and frames.dtype == np.uint8
    assert actions.shape == (N, len(BUTTONS))
    assert ram.shape == (N, RAM_SIZE)
    assert palette.shape == (NUM_COLORS, 3)

    # The episodes have to tile the arrays exactly, with no gap and no overlap, or a
    # training window could run from the end of one level into the start of another.
    assert episodes["start"][0] == 0
    assert np.array_equal(episodes["start"][1:], episodes["start"][:-1] + episodes["length"][:-1])
    assert episodes["start"][-1] + episodes["length"][-1] == N

    # The converted arrays are checked against the original files by a different route
    # than the converter took: PIL resolves the colours itself here, so a wrong palette
    # offset or a shifted crop would show up as a mismatch.
    scanned = scan()
    assert len(scanned) == len(episodes)

    rng = np.random.default_rng(0)

    for _ in range(NUM_SAMPLES):
        e = rng.integers(len(episodes))
        i = rng.integers(episodes["length"][e])
        index = episodes["start"][e] + i

        name, world, stage, win, paths = scanned[e]
        path = paths[i]

        rgb = np.asarray(Image.open(path).convert("RGB"))[CROP_TOP:CROP_TOP + FRAME_H]
        assert np.array_equal(palette[frames[index]], rgb)

        byte = int(NAME.match(os.path.basename(path)).group(7))
        assert np.array_equal(actions[index], to_action(byte))

        with open(path, "rb") as f:
            assert np.array_equal(ram[index], read_ram(f.read()))

        assert (episodes["world"][e], episodes["stage"][e], episodes["win"][e]) == (world, stage, win)

    print(f"{NUM_SAMPLES} random frames match the original PNGs: pixels, buttons and RAM")

    all_actions = np.asarray(actions)
    left, right, up, down = [all_actions[:, BUTTONS.index(name)] for name in ("LEFT", "RIGHT", "UP", "DOWN")]

    assert set(np.unique(all_actions).tolist()) == {0, 1}
    assert not np.any(left & right)
    assert not np.any(up & down)

    # The RAM addresses in envs/mario.py were written from the published RAM map. Here
    # they are checked against labels that come from somewhere else: the world and stage
    # in each folder name, and whether the run was recorded as a death.
    state = decode(np.asarray(ram))
    minutes = np.zeros(8)

    for episode in episodes:
        start = episode["start"]
        end = start + episode["length"]

        assert np.all(state["world"][start:end] == episode["world"])
        assert np.all(state["stage"][start:end] == episode["stage"])

        if (not episode["win"]):
            assert state["player_state"][end - 1] == DEAD

        minutes[episode["world"]] += episode["length"] / FPS / 60

    print("RAM world and stage match the folder names on every frame; every failed run ends dead")

    held = {name: round(float(all_actions[:, i].mean()), 3) for i, name in enumerate(BUTTONS)}
    print(f"fraction of frames each button is held: {held}")
    print(f"minutes of play per world: {[round(float(m), 1) for m in minutes]}  total {minutes.sum():.0f}")
    print(f"episodes: {int(episodes['win'].sum())} wins, {int((~episodes['win']).sum())} deaths")

    print("\nAll tests passed.")

if (__name__ == "__main__"):
    main()
