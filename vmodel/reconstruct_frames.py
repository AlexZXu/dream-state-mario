from vmodel.tokenizer import Tokenizer, TokenizerProps, colorize
from vmodel.train_tokenizer import get_device, CHECKPOINT
from envs.smb import load
from envs.smb_frames import SmbFrames
import os

import numpy as np
import torch
from PIL import Image

FRAMES_PER_WORLD = 200
BATCH_SIZE = 50
OUT_DIR = "data/samples"


@torch.no_grad()
def reconstruct(tokenizer, frames, palette):
    # frames shape: (batch, 224, 256) colour numbers -> the same, after a round trip
    # through the latent. The mean is decoded, as the dynamics model will do.
    mu, logvar = tokenizer.encode(colorize(frames, palette))

    return tokenizer.decode(mu).argmax(dim=1)


def main():
    device = get_device()

    tokenizer = Tokenizer(TokenizerProps()).to(device)
    tokenizer.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    tokenizer.eval()

    val_dset = SmbFrames(train=False)
    palette_rgb = load()["palette"]
    palette = torch.from_numpy(palette_rgb).float().to(device) / 255.0

    rng = np.random.default_rng(0)

    correct = 0
    total = 0
    exact_frames = 0
    num_frames = 0
    sheet = []

    for world in range(8):
        # The held-out set is half world 8, so the worlds are scored separately. An
        # average over all frames would hide a world the tokenizer handles badly.
        candidates = np.flatnonzero(val_dset.world == world)
        chosen = rng.choice(candidates, size=min(FRAMES_PER_WORLD, len(candidates)), replace=False)

        world_correct = 0
        world_total = 0
        worst = None

        for i in range(0, len(chosen), BATCH_SIZE):
            frames = torch.stack([val_dset[j] for j in chosen[i:i + BATCH_SIZE]]).to(device)
            recon = reconstruct(tokenizer, frames, palette)

            wrong = (recon != frames)
            wrong_per_frame = wrong.flatten(1).sum(dim=1)

            world_correct += (~wrong).sum().item()
            world_total += wrong.numel()
            exact_frames += (wrong_per_frame == 0).sum().item()
            num_frames += len(frames)

            k = wrong_per_frame.argmax().item()

            if (worst is None or wrong_per_frame[k].item() > worst[0]):
                worst = (wrong_per_frame[k].item(), frames[k].cpu().numpy(), recon[k].cpu().numpy())

        print(f"world {world + 1}  {world_correct / world_total * 100:.3f}% of pixels exact  worst frame {worst[0]} wrong pixels")

        correct += world_correct
        total += world_total

        # One row per world, showing the frame it did worst on: original, reconstruction,
        # and the wrong pixels in white on black.
        errors = np.where(worst[1] != worst[2], 255, 0).astype(np.uint8)
        errors = np.repeat(errors[:, :, None], 3, axis=2)
        sheet.append(np.concatenate([palette_rgb[worst[1]], palette_rgb[worst[2]], errors], axis=1))

    print(f"\nall worlds  {correct / total * 100:.3f}% of pixels exact  {exact_frames}/{num_frames} frames with every pixel right")

    os.makedirs(OUT_DIR, exist_ok=True)
    Image.fromarray(np.concatenate(sheet, axis=0)).save(f"{OUT_DIR}/tokenizer.png")
    print(f"saved {OUT_DIR}/tokenizer.png  (original | reconstruction | wrong pixels, worst frame of each world)")

if (__name__ == "__main__"):
    main()
