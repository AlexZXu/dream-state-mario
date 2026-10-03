from mmodel.flow import sample
from mmodel.train_dynamics import load_denoiser, CHECKPOINT as DYNAMICS_CHECKPOINT
from vmodel.tokenizer import Tokenizer, TokenizerProps
from vmodel.train_tokenizer import CHECKPOINT as TOKENIZER_CHECKPOINT
from envs.smb import load
from envs.smb_sequences import SmbSequences, FRAME_SKIP

import numpy as np
import torch

NUM_STEPS = 4  # Euler steps per generated frame
TAU_CTX = 0.1  # noise put on the context at play time, inside the range it was trained on


class Dreamer:
    """The game with no game engine: a tokenizer decoder, a dynamics model and a window of
    the last few latents. seed() starts it from a real moment, step() plays one step."""

    def __init__(self, device, num_steps=NUM_STEPS, tau_ctx=TAU_CTX):
        self.device = device
        self.num_steps = num_steps
        self.tau_ctx = tau_ctx

        self.tokenizer = Tokenizer(TokenizerProps()).to(device)
        self.tokenizer.load_state_dict(torch.load(TOKENIZER_CHECKPOINT, map_location=device))
        self.tokenizer.eval()

        self.denoiser = load_denoiser(DYNAMICS_CHECKPOINT, device)

        # The held-out episodes supply the starting points, so a seed is never a moment
        # the dynamics model trained on.
        self.dset = SmbSequences(train=False)
        self.palette = load()["palette"]

        self.context = None  # (1, L, C, 28, 32) normalized latents
        self.actions = None  # (1, L, 18) buttons that led into each context frame

        self.level = None
        self.x_pos = None
        self.powerup = None
        self.state_mask = torch.zeros(1, device=device)

    def seed(self, idx):
        """Load the context before held-out target idx. Returns the dataset's frame index of
        the frame the next step() will generate, so a caller can compare against the truth."""
        item = self.dset[idx]

        self.context = item["context"][None].to(self.device)
        self.actions = item["actions"][None, :-1].to(self.device)

        self.level = item["level"][None].to(self.device)
        self.x_pos = item["x_pos"][None].to(self.device)
        self.powerup = item["powerup"][None].to(self.device)

        return int(self.dset.index[idx])

    def use_state(self, on):
        # Until the reader of step 4 tracks x_pos through a rollout, the state can only be
        # switched on for the frozen values of the seed.
        self.state_mask = torch.full((1,), 1.0 if (on) else 0.0, device=self.device)

    @torch.no_grad()
    def step(self, action):
        # action: (6,) buttons held for the whole step, or (18,) the three per-frame readings
        action = np.asarray(action, dtype=np.float32)

        if (action.shape[0] == 6):
            action = np.tile(action, FRAME_SKIP)

        actions = torch.cat([self.actions, torch.from_numpy(action)[None, None].to(self.device)], dim=1)

        cond_inputs = {
            "actions": actions,
            "level": self.level,
            "x_pos": self.x_pos,
            "powerup": self.powerup,
            "state_mask": self.state_mask
        }

        z = sample(self.denoiser, self.context, cond_inputs, self.num_steps, self.tau_ctx)

        self.context = torch.cat([self.context[:, 1:], z[:, None]], dim=1)
        self.actions = actions[:, 1:]

        return self.decode(z)

    @torch.no_grad()
    def decode(self, z):
        # z shape: (1, C, 28, 32) normalized -> (224, 256) colour numbers
        logits = self.tokenizer.decode(self.dset.denormalize(z.cpu()).to(self.device))

        return logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    def rgb(self, frame):
        return self.palette[frame]
