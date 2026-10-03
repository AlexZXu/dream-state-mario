from mmodel.dreamer import Dreamer
from vmodel.train_tokenizer import get_device
from envs.smb import load
from envs.smb_sequences import FRAME_SKIP
import os

import numpy as np
import torch
from PIL import Image

NUM_ROLLOUTS = 16
HORIZON = 100  # steps, 5 seconds
REPORT_AT = [1, 2, 5, 10, 20, 40, 60, 100]
OUT_DIR = "data/samples"


def main():
    device = get_device()
    dreamer = Dreamer(device)

    data = load()
    frames = data["frames"]
    actions = data["actions"]
    episodes = data["episodes"]

    rng = np.random.default_rng(0)

    # Held-out targets with HORIZON more steps left in their episode.
    end = episodes["start"] + episodes["length"]
    candidates = []

    for idx in rng.permutation(len(dreamer.dset)):
        t = dreamer.dset.index[idx]
        e = np.searchsorted(episodes["start"], t, side="right") - 1

        if (t + FRAME_SKIP * HORIZON < end[e]):
            candidates.append(idx)

        if (len(candidates) == NUM_ROLLOUTS):
            break

    # Three numbers per horizon. "generated" is the model fed the real buttons from a
    # real start. "frozen" is the last context frame held still, which is what a model
    # that learned nothing about motion would score. "tokenizer" is the real frame after
    # a round trip through the tokenizer, the best the decoder allows.
    generated = np.zeros((NUM_ROLLOUTS, HORIZON))
    frozen = np.zeros((NUM_ROLLOUTS, HORIZON))
    tokenizer = np.zeros((NUM_ROLLOUTS, HORIZON))
    strip = None

    for r, idx in enumerate(candidates):
        t = dreamer.seed(idx)
        last_context = np.asarray(frames[t - FRAME_SKIP])
        real_row = []
        dream_row = []

        for h in range(HORIZON):
            f = t + FRAME_SKIP * h
            real = np.asarray(frames[f])

            # the buttons of the three frames leading into frame f, as recorded
            dream = dreamer.step(np.asarray(actions[f - FRAME_SKIP + 1:f + 1]).reshape(-1))

            z = dreamer.dset.normalize(torch.from_numpy(np.asarray(dreamer.dset.latents[f]).astype(np.float32)))
            round_trip = dreamer.decode(z[None].to(device))

            generated[r, h] = (dream == real).mean()
            frozen[r, h] = (last_context == real).mean()
            tokenizer[r, h] = (round_trip == real).mean()

            if (r == 0 and h + 1 in REPORT_AT):
                real_row.append(dreamer.rgb(real))
                dream_row.append(dreamer.rgb(dream))

        if (r == 0):
            strip = np.concatenate([np.concatenate(real_row, axis=1), np.concatenate(dream_row, axis=1)], axis=0)

        print(f"rollout {r + 1}/{NUM_ROLLOUTS}  pixels exact after {HORIZON} steps: {generated[r, -1] * 100:.1f}%")

    print(f"\n{'steps':>6} {'generated':>10} {'frozen':>8} {'tokenizer':>10}")

    for h in REPORT_AT:
        print(f"{h:>6} {generated[:, h - 1].mean() * 100:>9.1f}% {frozen[:, h - 1].mean() * 100:>7.1f}% {tokenizer[:, h - 1].mean() * 100:>9.1f}%")

    os.makedirs(OUT_DIR, exist_ok=True)
    Image.fromarray(strip.astype(np.uint8)).save(f"{OUT_DIR}/rollout.png")
    print(f"\nsaved {OUT_DIR}/rollout.png  (top: real, bottom: generated, at steps {REPORT_AT})")

if (__name__ == "__main__"):
    main()
