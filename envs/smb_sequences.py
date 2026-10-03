from envs.mario import decode, BUTTONS
from envs.smb import load, DATA_DIR
from envs.smb_frames import split
import numpy as np
import torch
from torch.utils.data import Dataset

FRAME_SKIP = 3  # recorded at 60 fps, modelled at 20
CONTEXT_FRAMES = 8
NUM_STAGES = 4
X_POS_SCALE = 8192.0  # the longest level is under 7000 pixels, so x_pos / scale stays in [0, 1)

LATENTS = f"{DATA_DIR}/latents.npy"
LATENT_STATS = f"{DATA_DIR}/latent_stats.npy"


def level_id(world, stage):
    return world * NUM_STAGES + stage


class SmbSequences(Dataset):
    def __init__(self, train=True, context_frames=CONTEXT_FRAMES, worlds=None, latents=None):
        data = load()
        train_episodes, val_episodes = split(data["episodes"])
        chosen = data["episodes"][train_episodes if (train) else val_episodes]

        if (worlds is not None):
            chosen = chosen[np.isin(chosen["world"], worlds)]

        self.context_frames = context_frames
        self.actions = data["actions"]
        self.ram = data["ram"]

        # latents shape: (N, C, 28, 32) float16, one per recorded frame, memory-mapped
        self.latents = np.load(LATENTS, mmap_mode="r") if (latents is None) else latents

        stats = np.load(LATENT_STATS) if (latents is None) else np.stack([np.zeros(self.latents.shape[1]), np.ones(self.latents.shape[1])])
        self.latent_mean = torch.from_numpy(stats[0]).float()[:, None, None]
        self.latent_std = torch.from_numpy(stats[1]).float()[:, None, None]

        # A target needs context_frames earlier frames, FRAME_SKIP apart, inside the same
        # episode, and the oldest of those needs its FRAME_SKIP - 1 lead-in frames of
        # buttons too, so the first frames of each episode are never targets. Every later
        # frame is, so the three ways of laying a 20 fps grid over the 60 fps recording
        # all appear.
        span = FRAME_SKIP * context_frames + FRAME_SKIP - 1

        self.index = np.concatenate([
            np.arange(episode["start"] + span, episode["start"] + episode["length"])
            for episode in chosen
        ])
        self.level = np.concatenate([
            np.full(episode["length"] - span, level_id(episode["world"], episode["stage"]))
            for episode in chosen
        ])

    def __len__(self):
        return self.index.shape[0]

    def normalize(self, z):
        # z shape: (..., C, 28, 32) -> zero mean, unit std per channel, the scale the
        # diffusion noise is drawn at
        return (z - self.latent_mean) / self.latent_std

    def denormalize(self, z):
        return z * self.latent_std + self.latent_mean

    def __getitem__(self, idx):
        t = self.index[idx]
        L = self.context_frames

        # frames shape: (L + 1,) -- the context, oldest first, then the target
        frames = np.arange(t - FRAME_SKIP * L, t + 1, FRAME_SKIP)

        latents = torch.from_numpy(np.asarray(self.latents[frames]).astype(np.float32))
        latents = self.normalize(latents)

        # The buttons held during the FRAME_SKIP frames that led up to each frame, kept
        # separately rather than merged: about 9% of steps have a button change inside
        # them, and the website will just repeat its one reading three times.
        # actions shape: (L + 1, FRAME_SKIP * 6)
        actions = np.stack([self.actions[frame - FRAME_SKIP + 1:frame + 1].reshape(-1) for frame in frames])

        state = decode(np.asarray(self.ram[t]))

        return {
            "context": latents[:-1],
            "target": latents[-1],
            "actions": torch.from_numpy(actions).float(),
            "level": torch.tensor(self.level[idx], dtype=torch.long),
            "x_pos": torch.tensor(state["x_pos"] / X_POS_SCALE, dtype=torch.float32),
            "powerup": torch.tensor(state["powerup"], dtype=torch.long)
        }
