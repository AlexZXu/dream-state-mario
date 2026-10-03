from vmodel.tokenizer import Tokenizer, TokenizerProps, colorize
from vmodel.train_tokenizer import get_device, CHECKPOINT
from envs.smb import load
from envs.smb_frames import split
from envs.smb_sequences import LATENTS, LATENT_STATS
import os
import time

import numpy as np
import torch

BATCH_SIZE = 256


@torch.no_grad()
def main():
    device = get_device()

    tokenizer = Tokenizer(TokenizerProps()).to(device)
    tokenizer.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    tokenizer.eval()

    data = load()
    frames = data["frames"]
    palette = torch.from_numpy(data["palette"]).float().to(device) / 255.0

    N = frames.shape[0]
    C = tokenizer.props.latent_channels

    # The posterior mean is stored, not a sample: the dynamics model should learn to
    # predict what the frame is, and the decoder was already trained to put up with the
    # noise. float16 keeps 737k latents at 10.6 GB.
    if (os.path.exists(LATENTS)):
        latents = np.load(LATENTS, mmap_mode="r+")
        assert latents.shape == (N, C, 28, 32)
    else:
        latents = np.lib.format.open_memmap(LATENTS, mode="w+", dtype=np.float16, shape=(N, C, 28, 32))

    start = time.time()

    for i in range(0, N, BATCH_SIZE):
        # The first run was killed 70% of the way through, so the script picks up where
        # one left off. The file starts as zeros and a real latent is never all zero, so
        # a batch with no zero rows is already done. Delete the file to force a re-encode,
        # which is needed whenever the tokenizer is retrained.
        if (np.asarray(latents[i:i + BATCH_SIZE]).any(axis=(1, 2, 3)).all()):
            continue

        batch = torch.from_numpy(np.asarray(frames[i:i + BATCH_SIZE])).long().to(device)
        mu, logvar = tokenizer.encode(colorize(batch, palette))

        latents[i:i + BATCH_SIZE] = mu.cpu().numpy().astype(np.float16)

        if ((i // BATCH_SIZE) % 200 == 0):
            print(f"frame {i}/{N}  {time.time() - start:.0f}s")

    latents.flush()

    # Per-channel mean and std over the training episodes only, so the validation set
    # does not leak into the scale the noise is drawn at.
    train_episodes, val_episodes = split(data["episodes"])
    picked = np.concatenate([
        np.arange(episode["start"], episode["start"] + episode["length"], 50)
        for episode in data["episodes"][train_episodes]
    ])

    sample = np.asarray(latents[picked]).astype(np.float32)
    stats = np.stack([sample.mean(axis=(0, 2, 3)), sample.std(axis=(0, 2, 3))])
    np.save(LATENT_STATS, stats)

    print(f"done in {time.time() - start:.0f}s  latent mean {np.round(stats[0], 2).tolist()}  std {np.round(stats[1], 2).tolist()}")

if (__name__ == "__main__"):
    main()
