from simul.engine import Game
from simul.run_replica import Bot, buttons
from simul.redream import Redream
from mmodel.dreamer import Dreamer
from vmodel.train_tokenizer import get_device
import os

import numpy as np
import torch
from PIL import Image

SAMPLES = "data/samples/redream.png"
LEVEL = 16  # 5-1: enemies all the way, and the bot runs 1,400 pixels into it without dying
STEPS = 200
SHOWN = (10, 50, 100, 150, 199)

# (name, tau_guide, anchor). tau_guide 1 ignores the simulation's frame, 0 copies it.
CONFIGS = [
    ("model alone", 1.0, False),
    ("true history", 1.0, True),
    ("guide 0.8", 0.8, False),
    ("guide 0.6", 0.6, False),
    ("guide 0.4", 0.4, False),
    ("guide 0.6 + true history", 0.6, True)
]


def record(level, steps):
    # One run of the simulation, kept so every config is asked to follow the same game.
    game = Game(level)
    bot = Bot()
    trace = []

    first = game.render()

    for _ in range(steps):
        action = bot(game)
        frame = game.step(action)
        x, y, width, height = game.boxes[-1]

        trace.append({
            "action": action, "frame": frame, "cut": game.cut, "level": game.level,
            "x_pos": game.x, "powerup": 2 if (game.fire) else int(game.big),
            "mario": (slice(max(y, 0), max(y + height, 0)), slice(max(x, 0), max(x + width, 0)))
        })

    return first, trace


def replay(dreamer, first, trace, tau_guide, anchor):
    torch.manual_seed(0)
    dreamer.seed_frames(first, trace[0]["level"])
    dreamer.use_state(True)

    frames = []

    for step in trace:
        # A respawn or a new level is a fresh start for the model too.
        if (step["cut"]):
            dreamer.seed_frames(step["frame"], step["level"], step["x_pos"], step["powerup"])

        dreamer.set_state(step["level"], step["x_pos"], step["powerup"])
        frames.append(dreamer.step(step["action"], guide=step["frame"], tau_guide=tau_guide, anchor=anchor))

    return np.stack(frames)


def main():
    dreamer = Dreamer(get_device())

    # The encoder and decoder alone, on a simulation frame: the ceiling for everything
    # below, and a check that the simulation's frames look like the game to the tokenizer.
    first, trace = record(LEVEL, STEPS)
    exact = np.stack([step["frame"] for step in trace])

    round_trip = dreamer.decode(dreamer.encode(exact[100]))
    ceiling = (round_trip == exact[100]).mean()
    assert ceiling > 0.98, ceiling
    print(f"tokenizer round trip on a simulation frame: {ceiling:.2%} of pixels exact\n")

    rows = [np.concatenate(list(exact[list(SHOWN)]), axis=1)]
    print(f"{'':28s}  pixels matching the simulation    Mario's box")
    print(f"{'':28s}  steps 1-20  81-100  181-200       steps 1-20  81-100  181-200")

    for name, tau_guide, anchor in CONFIGS:
        frames = replay(dreamer, first, trace, tau_guide, anchor)

        assert frames.shape == exact.shape and frames.max() < 64

        same = (frames == exact)
        overall = same.mean(axis=(1, 2))
        mario = np.array([same[i][step["mario"]].mean() if (same[i][step["mario"]].size > 0) else np.nan for i, step in enumerate(trace)])

        spans = [slice(0, 20), slice(80, 100), slice(180, 200)]
        print(
            f"{name:28s}  " + "  ".join(f"{overall[span].mean():7.1%}" for span in spans)
            + "        " + "  ".join(f"{np.nanmean(mario[span]):7.1%}" for span in spans)
        )

        rows.append(np.concatenate(list(frames[list(SHOWN)]), axis=1))

    # Standing still is where the prototype erases Mario. The player's rules for it:
    # the exact frame until the first button, then true history while he is not moving.
    torch.manual_seed(0)
    redream = Redream(Game(0), dreamer)
    kept = []

    for step in range(160):
        # 40 steps untouched, a short walk, then 90 steps standing still again
        frame, dreamed = redream.step(buttons("RIGHT") if (40 <= step < 60) else buttons())
        x, y, width, height = redream.game.boxes[-1]
        kept.append((dreamed[y + 16:y + 32, x:x + width] == frame[y + 16:y + 32, x:x + width]).mean())

        if (step < 40):
            assert np.array_equal(dreamed, frame)

    assert min(kept) > 0.5 and np.mean(kept[40:60]) > 0.65 and np.mean(kept[70:]) > 0.75, (min(kept), np.mean(kept[40:60]), np.mean(kept[70:]))
    print(
        f"\nstanding still: Mario exact before the first button, {np.mean(kept[40:60]):.0%} of his pixels kept while setting off, "
        f"{np.mean(kept[70:]):.0%} after stopping (worst step {min(kept):.0%})"
    )

    os.makedirs(os.path.dirname(SAMPLES), exist_ok=True)
    Image.fromarray(dreamer.palette[np.concatenate(rows, axis=0)]).save(SAMPLES)
    print(f"\nwrote {SAMPLES}: the simulation on top, then one row per line above; steps {SHOWN}")

    print("\nAll tests passed.")

if (__name__ == "__main__"):
    main()
