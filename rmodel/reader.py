import torch
import torch.nn as nn
import torch.nn.functional as F

DIED_WEIGHT = torch.tensor(5.0)  # how much a missed death costs against a false alarm


class ReaderProps:
    def __init__(self, args: dict = {}):
        self.latent_channels = args.get("latent_channels", 8)
        self.channels = args.get("channels", (64, 128, 128))
        self.num_powerups = args.get("num_powerups", 3)  # small, big, fire


class Reader(nn.Module):
    def __init__(self, props: ReaderProps):
        super(Reader, self).__init__()

        self.props = props
        c1, c2, c3 = props.channels

        # The two latents go in stacked as channels, the same way the denoiser takes its
        # context, so a shift between them is visible to a single conv.
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=props.latent_channels * 2, out_channels=c1, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=c2, out_channels=c3, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=c3, out_channels=c3, kernel_size=3, padding=1),
            nn.SiLU()
        )

        # Both an average and a max over the screen. The first version only averaged and
        # found 25% of deaths: a death is one small sprite (or its absence) in a scene
        # that is otherwise unchanged, and an average over 56 positions buries it.
        self.fc_dx = nn.Linear(in_features=c3 * 2, out_features=1)
        self.fc_died = nn.Linear(in_features=c3 * 2, out_features=1)
        self.fc_clear = nn.Linear(in_features=c3 * 2, out_features=1)
        self.fc_powerup = nn.Linear(in_features=c3 * 2, out_features=props.num_powerups)

    def forward(self, pair):
        # pair shape: (batch, 2, C, 28, 32) -- the previous latent, then the current one
        features = self.features(pair.flatten(1, 2))
        features = torch.cat([features.mean(dim=(2, 3)), features.amax(dim=(2, 3))], dim=1)

        # dx is in units of MAX_DX pixels; died and clear are logits
        return {
            "dx": self.fc_dx(features).squeeze(-1),
            "died": self.fc_died(features).squeeze(-1),
            "clear": self.fc_clear(features).squeeze(-1),
            "powerup": self.fc_powerup(features)
        }


def reader_loss(output, batch):
    # Steps where Mario was teleported have no meaningful dx, so they are left out of
    # the regression but still count for the other three heads.
    dx_error = (output["dx"] - batch["dx"]) ** 2 * batch["dx_valid"]
    dx_loss = dx_error.sum() / batch["dx_valid"].sum().clamp(min=1)

    # Only 4% of frames are deaths, and the unweighted loss settled on never raising a
    # false alarm while missing three deaths in four. A missed death strands the player
    # in a level with no Mario, which is the worse mistake, so positives count for more.
    died_loss = F.binary_cross_entropy_with_logits(output["died"], batch["died"], pos_weight=DIED_WEIGHT)
    clear_loss = F.binary_cross_entropy_with_logits(output["clear"], batch["clear"])
    powerup_loss = F.cross_entropy(output["powerup"], batch["powerup"])

    # dx is a fraction of 16 pixels, so its squared error is tiny next to the
    # classification terms; the weight puts an error of one pixel on the same footing.
    return 100 * dx_loss + died_loss + clear_loss + powerup_loss
