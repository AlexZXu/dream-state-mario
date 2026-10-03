import torch
import torch.nn as nn
import torch.nn.functional as F

FRAME_H = 224  # the NES draws 240 rows; the top and bottom 8 are overscan and get cropped
FRAME_W = 256
DOWNSAMPLE = 8  # 256x224 frame -> 32x28 latent
NUM_COLORS = 64  # the NES master palette; every pixel of a frame is one of these


class TokenizerProps:
    def __init__(self, args: dict = {}):
        self.in_channels = args.get("in_channels", 3)
        self.out_channels = args.get("out_channels", NUM_COLORS)
        self.latent_channels = args.get("latent_channels", 8)

        self.enc_channels = args.get("enc_channels", (64, 128, 256))
        self.dec_channels = args.get("dec_channels", (128, 64, 32))
        self.num_groups = args.get("num_groups", 8)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups):
        super(ResBlock, self).__init__()

        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)

        if (in_channels == out_channels):
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, X):
        h = self.conv1(F.silu(self.norm1(X)))
        h = self.conv2(F.silu(self.norm2(h)))

        return self.skip(X) + h


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        # The conv runs at the low resolution and PixelShuffle folds 4x the channels into
        # 2x the height and width, so the upsample costs a quarter of a conv at the
        # output resolution. The decoder is the only part that touches full-size frames
        # in the browser, so this is where the frame budget goes.
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels * 4, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, X):
        return self.shuffle(self.conv(X))


class Encoder(nn.Module):
    def __init__(self, props: TokenizerProps):
        super(Encoder, self).__init__()

        c1, c2, c3 = props.enc_channels

        # The first conv already strides, so no ResBlock ever runs at 256x224. The encoder
        # is training-only, but a full-resolution block would dominate the step time.
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=props.in_channels, out_channels=c1, kernel_size=3, stride=2, padding=1),
            ResBlock(c1, c1, props.num_groups),
            nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=3, stride=2, padding=1),
            ResBlock(c2, c2, props.num_groups),
            nn.Conv2d(in_channels=c2, out_channels=c3, kernel_size=3, stride=2, padding=1),
            ResBlock(c3, c3, props.num_groups),
            ResBlock(c3, c3, props.num_groups),
            nn.GroupNorm(num_groups=props.num_groups, num_channels=c3),
            nn.SiLU()
        )

        self.conv_mu = nn.Conv2d(in_channels=c3, out_channels=props.latent_channels, kernel_size=3, padding=1)
        self.conv_logvar = nn.Conv2d(in_channels=c3, out_channels=props.latent_channels, kernel_size=3, padding=1)

    def forward(self, X):
        # X shape: (batch, 3, 224, 256)
        features = self.layers(X)

        # mu, logvar shape: (batch, latent_channels, 28, 32)
        return self.conv_mu(features), self.conv_logvar(features)


class Decoder(nn.Module):
    def __init__(self, props: TokenizerProps):
        super(Decoder, self).__init__()

        c1, c2, c3 = props.dec_channels

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=props.latent_channels, out_channels=c1, kernel_size=3, padding=1),
            ResBlock(c1, c1, props.num_groups),
            ResBlock(c1, c1, props.num_groups),
            Upsample(c1, c2),
            ResBlock(c2, c2, props.num_groups),
            Upsample(c2, c3),
            ResBlock(c3, c3, props.num_groups),
            nn.GroupNorm(num_groups=props.num_groups, num_channels=c3),
            nn.SiLU(),
            Upsample(c3, props.out_channels)
        )

    def forward(self, z):
        # z shape: (batch, latent_channels, 28, 32) -> (batch, out_channels, 224, 256)
        return self.layers(z)


class Tokenizer(nn.Module):
    def __init__(self, props: TokenizerProps):
        super(Tokenizer, self).__init__()

        self.props = props

        self.encoder = Encoder(props)
        self.decoder = Decoder(props)

    def reparameterization(self, mu, logvar) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def encode(self, X):
        return self.encoder(X)

    def decode(self, z):
        # The output is left raw. out_channels is 3 for an RGB head and the palette size
        # for a per-pixel classification head, so the activation belongs to the loss.
        return self.decoder(z)

    def forward(self, X):
        # Unlike the CarRacing VAE there is no flatten: the latent keeps its 28x32 grid,
        # because the dynamics model needs to know where on screen things are.
        mu, logvar = self.encode(X)
        z = self.reparameterization(mu, logvar)

        X_recon = self.decode(z)

        return X_recon, mu, logvar


def colorize(frame, palette):
    # frame shape: (batch, h, w) int64 colour numbers; palette shape: (64, 3) in [0, 1]
    # -> (batch, 3, h, w). The encoder is given real colours rather than 64 one-hot
    # planes, so two shades of the same brick start out close together.
    return palette[frame].permute(0, 3, 1, 2)


def tokenizer_loss(logits, target, mu, logvar, kl_weight):
    # logits shape: (batch, 64, h, w); target shape: (batch, h, w) colour numbers
    batch_size = target.shape[0]

    # A frame is pixel art drawn from 64 fixed colours, so reconstruction is a 64-way
    # classification per pixel. A regression on RGB can answer with a blend of two
    # colours, which is exactly what blur is; a classifier has to commit to one.
    reconstruct_loss = F.cross_entropy(logits, target, reduction='sum') / batch_size

    # KL(q(z|x) || N(0, I)). Both terms are summed over their dimensions and averaged
    # over the batch, as in the CarRacing VAE, but the weight is tiny: the latent only has
    # to stay on a bounded scale for the dynamics model, not be sampled from the prior.
    kl_loss = 0.5 * torch.sum(mu**2 + torch.exp(logvar) - 1 - logvar) / batch_size

    return reconstruct_loss + kl_weight * kl_loss, reconstruct_loss, kl_loss


def pixel_accuracy(logits, target):
    correct = (logits.argmax(dim=1) == target).float().cpu()
    target = target.cpu()

    # Most of a frame is flat sky or black, which any model gets right. The second number
    # only counts pixels that are not the frame's most common colour, which is where
    # Mario, the enemies, the coins and the HUD digits are.
    background = torch.mode(target.flatten(1), dim=1).values[:, None, None]
    foreground = (target != background).float()

    return correct.mean().item(), ((correct * foreground).sum() / foreground.sum()).item()
