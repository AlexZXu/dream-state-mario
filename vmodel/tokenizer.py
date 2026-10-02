import torch
import torch.nn as nn
import torch.nn.functional as F

FRAME_H = 224  # the NES draws 240 rows; the top and bottom 8 are overscan and get cropped
FRAME_W = 256
DOWNSAMPLE = 8  # 256x224 frame -> 32x28 latent


class TokenizerProps:
    def __init__(self, args: dict = {}):
        self.in_channels = args.get("in_channels", 3)
        self.out_channels = args.get("out_channels", 3)
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
