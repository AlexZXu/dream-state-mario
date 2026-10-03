from envs.mario import decode, DYING, DEAD
from envs.smb import load
from envs.smb_frames import split
from envs.smb_sequences import FRAME_SKIP, LATENTS, LATENT_STATS
import numpy as np
import torch
from torch.utils.data import Dataset

FLAGPOLE = 0x04  # player state while sliding down the flagpole
LEVEL_END = 0x05  # player state while walking from the flagpole into the castle
RAM_Y_SCREEN = 0x00B5  # 1 while Mario is on the screen, more once he has fallen below it

MAX_DX = 16  # pixels per step; Mario's top running speed is about 8


def event_labels(ram, episodes):
    # ram shape: (N, 2048). Returns one label per recorded frame.
    state = decode(ram)
    player_state = state["player_state"]

    died = np.isin(player_state, [DYING, DEAD])

    # Falling into a pit has no death animation: the player state stays "playing" until
    # the single dead frame, and the only sign is Mario below the screen. That also
    # happens harmlessly inside pipes and bonus rooms, so it is only called a death for
    # the unbroken stretch that runs up to the end of a failed run. That stretch is
    # about 4 seconds: the game keeps going, with no Mario on screen, until the life ends.
    for episode in episodes[~episodes["win"]]:
        t = episode["start"] + episode["length"] - 1

        while (t >= episode["start"] and ram[t, RAM_Y_SCREEN] > 1):
            died[t] = True
            t -= 1

    # Castle levels end at the axe, with the player state still "playing" when the
    # recording stops, so they have no clear label here. FIX THIS LATER
    clear = np.isin(player_state, [FLAGPOLE, LEVEL_END])

    # How far Mario moved through the level over one model step. A pipe or a level
    # change teleports him, which is not movement, so those steps are masked out.
    dx = np.zeros(len(ram), dtype=np.float32)
    dx[FRAME_SKIP:] = state["x_pos"][FRAME_SKIP:] - state["x_pos"][:-FRAME_SKIP]
    dx_valid = (np.abs(dx) <= MAX_DX)

    return {
        "died": died,
        "clear": clear,
        "dx": np.clip(dx, -MAX_DX, MAX_DX) / MAX_DX,
        "dx_valid": dx_valid,
        "powerup": state["powerup"]
    }


class SmbEvents(Dataset):
    def __init__(self, train=True, latents=None):
        data = load()
        train_episodes, val_episodes = split(data["episodes"])
        chosen = data["episodes"][train_episodes if (train) else val_episodes]

        self.labels = event_labels(np.asarray(data["ram"]), data["episodes"])
        self.latents = np.load(LATENTS, mmap_mode="r") if (latents is None) else latents

        C = self.latents.shape[1]
        stats = np.load(LATENT_STATS) if (latents is None) else np.stack([np.zeros(C), np.ones(C)])
        self.latent_mean = torch.from_numpy(stats[0]).float()[:, None, None]
        self.latent_std = torch.from_numpy(stats[1]).float()[:, None, None]

        # The reader sees a frame and the one a step before it, so the first FRAME_SKIP
        # frames of an episode have no pair.
        self.index = np.concatenate([
            np.arange(episode["start"] + FRAME_SKIP, episode["start"] + episode["length"])
            for episode in chosen
        ])

    def __len__(self):
        return self.index.shape[0]

    def __getitem__(self, idx):
        t = self.index[idx]

        # pair shape: (2, C, 28, 32) -- the previous step's latent, then this one
        pair = torch.from_numpy(np.asarray(self.latents[[t - FRAME_SKIP, t]]).astype(np.float32))
        pair = (pair - self.latent_mean) / self.latent_std

        return {
            "pair": pair,
            "dx": torch.tensor(self.labels["dx"][t]),
            "dx_valid": torch.tensor(float(self.labels["dx_valid"][t])),
            "died": torch.tensor(float(self.labels["died"][t])),
            "clear": torch.tensor(float(self.labels["clear"][t])),
            "powerup": torch.tensor(self.labels["powerup"][t], dtype=torch.long)
        }
