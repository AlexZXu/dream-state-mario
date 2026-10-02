import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SCALAR_SCALE = 1000.0  # tau and x_pos arrive in [0, 1]; the sinusoid frequencies assume [0, 1000]


class DenoiserProps:
    def __init__(self, args: dict = {}):
        self.latent_channels = args.get("latent_channels", 8)
        self.context_frames = args.get("context_frames", 8)
        self.action_dim = args.get("action_dim", 6)  # left, right, up, down, A, B

        # 46M params. (128, 256, 512) is 68M and (96, 192, 384) is 40M; all three were timed
        # in the browser with random weights before any training, see export/export_onnx.py.
        self.channels = args.get("channels", (128, 256, 384))
        self.attention = args.get("attention", (False, True, True))
        self.blocks_per_level = args.get("blocks_per_level", 2)
        self.num_groups = args.get("num_groups", 32)
        self.num_heads = args.get("num_heads", 8)

        self.cond_dim = args.get("cond_dim", 512)
        self.num_levels = args.get("num_levels", 32)  # 8 worlds x 4 stages
        self.num_powerups = args.get("num_powerups", 3)  # small, big, fire


def scalar_embedding(x, dim):
    # x shape: (batch,) -> (batch, dim)
    half = dim // 2
    freqs = torch.exp(-math.log(10_000) * torch.arange(half, device=x.device) / half)
    angles = x[:, None] * SCALAR_SCALE * freqs[None, :]

    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class CondResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, props: DenoiserProps):
        super(CondResBlock, self).__init__()

        self.norm1 = nn.GroupNorm(num_groups=props.num_groups, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

        self.cond_proj = nn.Linear(in_features=props.cond_dim, out_features=out_channels * 2)

        self.norm2 = nn.GroupNorm(num_groups=props.num_groups, num_channels=out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)

        if (in_channels == out_channels):
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, X, cond):
        h = self.conv1(F.silu(self.norm1(X)))

        # Adaptive group norm: the conditioning sets the scale and shift of every channel
        # after normalization. The buttons and the noise level are global facts about the
        # frame, so they act on whole feature maps instead of being concatenated as pixels.
        scale, shift = self.cond_proj(cond)[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift

        h = self.conv2(F.silu(h))

        return self.skip(X) + h


class SelfAttention(nn.Module):
    def __init__(self, channels, props: DenoiserProps):
        super(SelfAttention, self).__init__()

        self.num_heads = props.num_heads
        self.d_head = channels // props.num_heads

        self.norm = nn.GroupNorm(num_groups=props.num_groups, num_channels=channels)
        self.qkv = nn.Linear(in_features=channels, out_features=channels * 3)
        self.out = nn.Linear(in_features=channels, out_features=channels)

    def forward(self, X):
        # X shape: (batch, channels, h, w)
        batch, channels, h, w = X.shape

        tokens = self.norm(X).flatten(2).transpose(1, 2)
        qkv = self.qkv(tokens).reshape(batch, h * w, 3, self.num_heads, self.d_head)

        # q, k, v shape: (batch, num_heads, h * w, d_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        weights = F.softmax(scores, dim=-1)

        attended = (weights @ v).transpose(1, 2).reshape(batch, h * w, channels)
        attended = self.out(attended).transpose(1, 2).reshape(batch, channels, h, w)

        return X + attended


class Stage(nn.Module):
    def __init__(self, in_channels, out_channels, attention, props: DenoiserProps):
        super(Stage, self).__init__()

        self.block = CondResBlock(in_channels, out_channels, props)
        self.attention = SelfAttention(out_channels, props) if (attention) else nn.Identity()

    def forward(self, X, cond):
        return self.attention(self.block(X, cond))


class Downsample(nn.Module):
    def __init__(self, channels):
        super(Downsample, self).__init__()

        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2, padding=1)

    def forward(self, X):
        return self.conv(X)


class Upsample(nn.Module):
    def __init__(self, channels):
        super(Upsample, self).__init__()

        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, X):
        return self.conv(F.interpolate(X, scale_factor=2.0, mode="nearest"))


class Denoiser(nn.Module):
    def __init__(self, props: DenoiserProps):
        super(Denoiser, self).__init__()

        self.props = props
        self.embed_dim = props.cond_dim // 4

        C = props.latent_channels
        L = props.context_frames
        cond_dim = props.cond_dim

        self.noise_mlp = nn.Sequential(
            nn.Linear(self.embed_dim * 2, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )

        self.action_mlp = nn.Sequential(
            nn.Linear((L + 1) * props.action_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )

        self.level_embedding = nn.Embedding(num_embeddings=props.num_levels, embedding_dim=self.embed_dim)
        self.powerup_embedding = nn.Embedding(num_embeddings=props.num_powerups, embedding_dim=self.embed_dim)
        self.state_mlp = nn.Sequential(
            nn.Linear(self.embed_dim * 3, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )

        # What the state embedding is replaced with when the state is dropped. Training
        # drops it at random so the same weights run with or without the harness
        # tracking level position, which is what makes the A/B in step 4 possible.
        self.null_state = nn.Parameter(torch.zeros(cond_dim))

        # The context frames go in as extra channels next to the noisy latent (frame
        # stacking). Scrolling is a small shift between frames, so a conv sees the same
        # tile in neighbouring channels and can copy it across without any attention.
        self.conv_in = nn.Conv2d(in_channels=C * (L + 1), out_channels=props.channels[0], kernel_size=3, padding=1)

        in_channels = props.channels[0]
        skip_channels = [in_channels]
        last = len(props.channels) - 1

        # note for important change -> ModuleList, not a plain list: a plain list is invisible
        # to nn.Module, so the blocks would be left out of parameters(), state_dict() and .to(device).
        self.down = nn.ModuleList()

        for i, channels in enumerate(props.channels):
            for _ in range(props.blocks_per_level):
                self.down.append(Stage(in_channels, channels, props.attention[i], props))
                in_channels = channels
                skip_channels.append(in_channels)

            if (i < last):
                self.down.append(Downsample(in_channels))
                skip_channels.append(in_channels)

        self.mid1 = Stage(in_channels, in_channels, True, props)
        self.mid2 = Stage(in_channels, in_channels, False, props)

        self.up = nn.ModuleList()

        for i, channels in reversed(list(enumerate(props.channels))):
            # One more stage than the way down, so every skip pushed there (including the
            # ones from conv_in and the downsamples) is consumed exactly once.
            for _ in range(props.blocks_per_level + 1):
                self.up.append(Stage(in_channels + skip_channels.pop(), channels, props.attention[i], props))
                in_channels = channels

            if (i > 0):
                self.up.append(Upsample(in_channels))

        self.norm_out = nn.GroupNorm(num_groups=props.num_groups, num_channels=in_channels)
        self.conv_out = nn.Conv2d(in_channels=in_channels, out_channels=C, kernel_size=3, padding=1)

        # The predicted velocity starts at exactly zero, so the first training steps are
        # not spent undoing a random output.
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def conditioning(self, tau, tau_ctx, actions, level, x_pos, powerup, state_mask):
        # tau, tau_ctx shape: (batch,) in [0, 1]
        # actions shape: (batch, L + 1, 6) -- buttons held going into each context frame,
        #                then the buttons held going into the frame being generated
        # level, powerup shape: (batch,) int64
        # x_pos, state_mask shape: (batch,) -- level progress in [0, 1], and 1.0 where the
        #                state is known or 0.0 where it was dropped
        noise = torch.cat([
            scalar_embedding(tau, self.embed_dim),
            scalar_embedding(tau_ctx, self.embed_dim)
        ], dim=-1)

        state = torch.cat([
            self.level_embedding(level),
            scalar_embedding(x_pos, self.embed_dim),
            self.powerup_embedding(powerup)
        ], dim=-1)

        state_mask = state_mask[:, None]
        state = state_mask * self.state_mlp(state) + (1 - state_mask) * self.null_state

        # cond shape: (batch, cond_dim)
        return self.noise_mlp(noise) + self.action_mlp(actions.flatten(1)) + state

    def denoise(self, z_noisy, context, cond):
        # z_noisy shape: (batch, C, 28, 32) -- the next latent at noise level tau
        # context shape: (batch, L, C, 28, 32) -- the previous latents, oldest first
        X = torch.cat([z_noisy, context.flatten(1, 2)], dim=1)
        X = self.conv_in(X)

        skips = [X]

        for layer in self.down:
            X = layer(X, cond) if (isinstance(layer, Stage)) else layer(X)
            skips.append(X)

        X = self.mid1(X, cond)
        X = self.mid2(X, cond)

        for layer in self.up:
            if (isinstance(layer, Stage)):
                X = layer(torch.cat([X, skips.pop()], dim=1), cond)
            else:
                X = layer(X)

        # velocity shape: (batch, C, 28, 32)
        return self.conv_out(F.silu(self.norm_out(X)))

    def forward(self, z_noisy, context, tau, tau_ctx, actions, level, x_pos, powerup, state_mask):
        # The two halves are separate methods because they are exported separately. In
        # half precision 1000 * tau is only known to the nearest 0.5, which scrambles the
        # high-frequency sinusoids, so conditioning() stays float32 in the browser and
        # only denoise() is converted.
        cond = self.conditioning(tau, tau_ctx, actions, level, x_pos, powerup, state_mask)

        return self.denoise(z_noisy, context, cond)
