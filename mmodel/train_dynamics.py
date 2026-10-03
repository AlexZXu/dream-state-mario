from mmodel.denoiser import Denoiser, DenoiserProps
from mmodel.flow import add_noise, sample_tau, flow_loss
from envs.smb_sequences import SmbSequences, FRAME_SKIP, CONTEXT_FRAMES
from vmodel.train_tokenizer import get_device
import copy
import math
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

STEPS = 30_000
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WARMUP_STEPS = 1000
EMA_DECAY = 0.999
CONTEXT_NOISE_MAX = 0.5  # context frames are noised to a random tau in [0, this] during training
STATE_DROPOUT = 0.25  # fraction of samples trained without level id, x_pos and power-up
GRAD_CLIP = 1.0
EVAL_EVERY = 2000
EVAL_BATCHES = 8
WORLDS = None  # e.g. [7] trains on world 8 only; None is every level
CHANNELS = (64, 128, 256)  # prototype width for the Mac; the DenoiserProps default is the full model
CHECKPOINT = "dynamics.pth"


def make_props():
    return DenoiserProps({
        "channels": CHANNELS,
        "context_frames": CONTEXT_FRAMES,
        "action_dim": FRAME_SKIP * 6
    })


def load_denoiser(path, device):
    """The EMA weights from a checkpoint, in eval mode, with the props they were trained with."""
    saved = torch.load(path, map_location=device)

    denoiser = Denoiser(DenoiserProps(saved["props"])).to(device)
    denoiser.load_state_dict(saved["ema"])
    denoiser.eval()

    return denoiser


def compute_loss(denoiser, batch, device):
    context = batch["context"].to(device)
    target = batch["target"].to(device)
    batch_size = target.shape[0]

    tau = sample_tau(batch_size, device)
    z_noisy, eps = add_noise(target, tau)

    # The context gets its own, milder noise. At play time the context is the model's
    # own earlier output, which is never exactly right, and a model that has only seen
    # perfect context treats every small error as fact and drifts.
    tau_ctx = torch.rand(batch_size, device=device) * CONTEXT_NOISE_MAX
    noisy_context, _ = add_noise(context.flatten(1, 2), tau_ctx)
    noisy_context = noisy_context.view_as(context)

    state_mask = (torch.rand(batch_size, device=device) >= STATE_DROPOUT).float()

    velocity = denoiser(
        z_noisy, noisy_context, tau, tau_ctx,
        batch["actions"].to(device), batch["level"].to(device),
        batch["x_pos"].to(device), batch["powerup"].to(device), state_mask
    )

    return flow_loss(velocity, target, eps)


@torch.no_grad()
def evaluate(ema, val_loader, device):
    losses = []

    for i, batch in enumerate(val_loader):
        if (i == EVAL_BATCHES):
            break

        losses.append(compute_loss(ema, batch, device).item())

    return sum(losses) / len(losses)


def batches(loader):
    # One pass over the data is about 20k batches, fewer than STEPS, so the loader is
    # walked again (reshuffled) for as long as the training loop keeps asking.
    while (True):
        for batch in loader:
            yield batch


def main():
    device = get_device()
    print(f"training on {device}")

    train_dset = SmbSequences(train=True, worlds=WORLDS)
    val_dset = SmbSequences(train=False, worlds=WORLDS)
    print(f"{len(train_dset)} training targets  {len(val_dset)} held-out targets")

    train_loader = DataLoader(train_dset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    props = make_props()
    denoiser = Denoiser(props).to(device)
    print(f"denoiser params {sum(param.numel() for param in denoiser.parameters()) / 1e6:.1f}M")

    # The EMA copy is what gets sampled from. Diffusion losses are noisy step to step,
    # and the average of the recent weights samples visibly better than the latest ones.
    ema = copy.deepcopy(denoiser).eval()

    for param in ema.parameters():
        param.requires_grad_(False)

    optimizer = optim.AdamW(params=denoiser.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # Linear warmup, then cosine down to zero.
    schedule = lambda step: min(1.0, (step + 1) / WARMUP_STEPS) * 0.5 * (1 + math.cos(math.pi * step / STEPS))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule)

    running_loss = 0.0
    start = time.time()

    for step, batch in enumerate(batches(train_loader)):
        if (step == STEPS):
            break

        optimizer.zero_grad()

        loss = compute_loss(denoiser, batch, device)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(denoiser.parameters(), max_norm=GRAD_CLIP)

        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            for ema_param, param in zip(ema.parameters(), denoiser.parameters()):
                ema_param.mul_(EMA_DECAY).add_(param, alpha=1 - EMA_DECAY)

        running_loss += loss.item()

        if (step % 100 == 99):
            print(f"step {step + 1}/{STEPS}  loss {running_loss / 100:.4f}  {time.time() - start:.0f}s")
            running_loss = 0.0

        if (step % EVAL_EVERY == EVAL_EVERY - 1):
            print(f"held-out loss (ema) {evaluate(ema, val_loader, device):.4f}")

            torch.save({"props": vars(props), "denoiser": denoiser.state_dict(), "ema": ema.state_dict()}, CHECKPOINT)

    torch.save({"props": vars(props), "denoiser": denoiser.state_dict(), "ema": ema.state_dict()}, CHECKPOINT)

if (__name__ == "__main__"):
    main()
