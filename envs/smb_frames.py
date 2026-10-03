from envs.smb import load
import numpy as np
import torch
from torch.utils.data import Dataset

VAL_EVERY = 10  # every tenth episode is held out


def split(episodes):
    # Whole episodes are held out, not random frames. At 60 fps a frame's neighbours are
    # nearly identical to it, so a random split would leave a copy of every validation
    # frame in the training set.
    index = np.arange(len(episodes))
    val = (index % VAL_EVERY == VAL_EVERY - 1)

    return index[~val], index[val]


class SmbFrames(Dataset):
    def __init__(self, train=True, crop=None):
        data = load()
        train_episodes, val_episodes = split(data["episodes"])
        chosen = data["episodes"][train_episodes if (train) else val_episodes]

        self.frames = data["frames"]
        self.crop = crop

        # The V model is trained on frames with no notion of which episode or what order
        # they came from, so the dataset is just the list of frame positions it may use.
        self.index = np.concatenate([
            np.arange(episode["start"], episode["start"] + episode["length"])
            for episode in chosen
        ])
        self.world = np.concatenate([
            np.full(episode["length"], episode["world"])
            for episode in chosen
        ])

    def __len__(self):
        return self.index.shape[0]

    def __getitem__(self, idx):
        # frame shape: (224, 256) uint8 colour numbers, read from disk on demand
        frame = self.frames[self.index[idx]]

        if (self.crop is not None):
            # The offset is a multiple of 8, so a patch sits on the same 8x8 grid as the
            # whole frame. The first version used any pixel offset, reasoning that the
            # tokenizer should see tiles at every alignment; but the game never scrolls
            # vertically and the horizontal scroll is already in the frames, so that
            # only made it learn 8x more alignments than it will ever be shown. On a
            # 64-frame memorisation test the aligned version reached 90% of
            # non-background pixels in 1500 steps against 70% for the unaligned one.
            top = 8 * np.random.randint(0, (frame.shape[0] - self.crop) // 8 + 1)
            left = 8 * np.random.randint(0, (frame.shape[1] - self.crop) // 8 + 1)
            frame = frame[top:top + self.crop, left:left + self.crop]

        return torch.from_numpy(np.array(frame)).long()
