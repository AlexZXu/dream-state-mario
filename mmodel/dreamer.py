from mmodel.flow import sample
from mmodel.train_dynamics import load_denoiser, CHECKPOINT as DYNAMICS_CHECKPOINT
from vmodel.tokenizer import Tokenizer, TokenizerProps, colorize
from vmodel.train_tokenizer import CHECKPOINT as TOKENIZER_CHECKPOINT
from envs.smb import load
from envs.smb_sequences import SmbSequences, FRAME_SKIP, CONTEXT_FRAMES, X_POS_SCALE

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

    def seed_frames(self, frame, level, x_pos=0, powerup=0):
        # Start from a frame that is not in the dataset: frame shape (224, 256) colour
        # numbers. The context is that one frame repeated with no buttons held, which is
        # what standing still at the start of a level looks like.
        z = self.encode(frame)

        self.context = z[:, None].repeat(1, CONTEXT_FRAMES, 1, 1, 1)
        self.actions = torch.zeros(1, CONTEXT_FRAMES, FRAME_SKIP * 6, device=self.device)
        self.set_state(level, x_pos, powerup)

    def set_state(self, level, x_pos, powerup):
        # x_pos in pixels into the level
        self.level = torch.tensor([level], dtype=torch.long, device=self.device)
        self.x_pos = torch.tensor([x_pos / X_POS_SCALE], dtype=torch.float32, device=self.device)
        self.powerup = torch.tensor([powerup], dtype=torch.long, device=self.device)

    @torch.no_grad()
    def encode(self, frame):
        # frame shape: (224, 256) colour numbers -> (1, C, 28, 32) normalized
        palette = torch.from_numpy(self.palette).float().to(self.device) / 255.0
        batch = torch.from_numpy(np.asarray(frame)).long()[None].to(self.device)
        mu, logvar = self.tokenizer.encode(colorize(batch, palette))

        return self.dset.normalize(mu.cpu()).to(self.device)

    def use_state(self, on):
        # Until the reader of step 4 tracks x_pos through a rollout, the state can only be
        # switched on for the frozen values of the seed.
        self.state_mask = torch.full((1,), 1.0 if (on) else 0.0, device=self.device)

    @torch.no_grad()
    def step(self, action, guide=None, tau_guide=1.0, anchor=False):
        # action: (6,) buttons held for the whole step, or (18,) the three per-frame readings
        # guide: optional (224, 256) frame of what the step should roughly look like (the
        # simulation's frame). Generation starts from it noised to tau_guide instead of
        # from pure noise. With anchor, the guide and not the generated frame goes into
        # the context, so the next step is predicted from true history and errors cannot
        # build up.
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

        z_guide = None if (guide is None) else self.encode(guide)

        if (z_guide is not None and tau_guide <= 0):
            z = z_guide
        else:
            z = sample(self.denoiser, self.context, cond_inputs, self.num_steps, self.tau_ctx, guide=z_guide, tau_guide=tau_guide)

        remembered = z_guide if (anchor and z_guide is not None) else z
        self.context = torch.cat([self.context[:, 1:], remembered[:, None]], dim=1)
        self.actions = actions[:, 1:]

        return self.decode(z)

    @torch.no_grad()
    def decode(self, z):
        # z shape: (1, C, 28, 32) normalized -> (224, 256) colour numbers
        logits = self.tokenizer.decode(self.dset.denormalize(z.cpu()).to(self.device))

        return logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    def rgb(self, frame):
        return self.palette[frame]
