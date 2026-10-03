from vmodel.tokenizer import Tokenizer, TokenizerProps, tokenizer_loss, pixel_accuracy, colorize
from envs.smb import load
from envs.smb_frames import SmbFrames
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

STEPS = 20_000
BATCH_SIZE = 32
CROP = 128  # the model is all convolutions, so it trains on patches and runs on whole frames
LEARNING_RATE = 3e-4
# 1e-4 and 1e-2 reached the same accuracy at 1250 steps, but 1e-2 keeps the posterior
# std near 0.1 instead of 0.005, so the decoder trains on latents that are a little off.
KL_WEIGHT = 1e-2
GRAD_CLIP = 10_000  # the loss is a sum over 16k pixels, so a typical gradient norm is about 5000
NOISE_LEVELS = [0.1, 0.2]  # latent noise, as a fraction of the latent's own spread, to test the decoder against
EVAL_EVERY = 2000
EVAL_FRAMES = 64
CHECKPOINT = "tokenizer.pth"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(tokenizer, frames, palette):
    tokenizer.eval()

    # The mean is decoded, not a sample: this is how the dynamics model will use the
    # tokenizer, and it keeps the number the same from one evaluation to the next.
    mu, logvar = tokenizer.encode(colorize(frames, palette))
    accuracy, foreground = pixel_accuracy(tokenizer.decode(mu), frames)

    # The dynamics model will never produce a latent exactly, so what matters as much as
    # the clean number is how fast the picture falls apart when the latent is a bit off.
    noisy = []

    for level in NOISE_LEVELS:
        z = mu + level * mu.std() * torch.randn_like(mu)
        noisy.append(pixel_accuracy(tokenizer.decode(z), frames)[1])

    tokenizer.train()

    return accuracy, foreground, noisy, mu.std().item(), torch.exp(0.5 * logvar).mean().item()


def main():
    device = get_device()
    print(f"training on {device}")

    train_dset = SmbFrames(train=True, crop=CROP)
    val_dset = SmbFrames(train=False)
    print(f"{len(train_dset)} training frames  {len(val_dset)} held-out frames")

    # The frames are memory-mapped, so worker processes would only add copying.
    data_loader = DataLoader(
        train_dset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    assert STEPS <= len(data_loader)

    # A fixed set of whole held-out frames spread across the episodes, so every
    # evaluation is on the same pictures and at the size the model will really run at.
    spread = torch.linspace(0, len(val_dset) - 1, EVAL_FRAMES).long()
    val_frames = torch.stack([val_dset[i] for i in spread]).to(device)

    palette = torch.from_numpy(load()["palette"]).float().to(device) / 255.0

    tokenizer = Tokenizer(TokenizerProps()).to(device)
    optimizer = optim.Adam(params=tokenizer.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS)

    running_loss = 0.0
    start = time.time()

    for step, frame in enumerate(data_loader):
        if (step == STEPS):
            break

        frame = frame.to(device)

        optimizer.zero_grad()

        X_recon, mu, logvar = tokenizer(colorize(frame, palette))

        loss, reconstruct_loss, kl_loss = tokenizer_loss(X_recon, frame, mu, logvar, KL_WEIGHT)
        loss.backward()

        # Without this the first run blew up at step 1050: one bad batch produced a huge
        # gradient, the reconstruction loss went from 2700 to 12000 in a hundred steps and
        # the latent scale never came back. Clipping at about twice the typical norm
        # leaves ordinary steps alone and caps the rare one.
        torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), max_norm=GRAD_CLIP)

        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        if (step % 100 == 99):
            print(
                f"step {step + 1}/{STEPS}  loss {running_loss / 100:.1f}  "
                f"recon {reconstruct_loss.item():.1f}  kl {kl_loss.item():.0f}  "
                f"{time.time() - start:.0f}s"
            )
            running_loss = 0.0

        if (step % EVAL_EVERY == EVAL_EVERY - 1):
            accuracy, foreground, noisy, latent_std, posterior_std = evaluate(tokenizer, val_frames, palette)
            print(
                f"held-out frames: {accuracy * 100:.3f}% of pixels exact  "
                f"{foreground * 100:.2f}% of non-background pixels  "
                f"with latent noise {NOISE_LEVELS}: {[round(value * 100, 2) for value in noisy]}  "
                f"latent std {latent_std:.2f}  posterior std {posterior_std:.3f}"
            )

            # The V model is trained on its own and then frozen, so the checkpoint is
            # written at every evaluation and the M model just loads it.
            torch.save(tokenizer.state_dict(), CHECKPOINT)

    torch.save(tokenizer.state_dict(), CHECKPOINT)

if (__name__ == "__main__"):
    main()
