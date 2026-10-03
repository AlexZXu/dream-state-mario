from mmodel.denoiser import Denoiser, DenoiserProps
from mmodel.flow import add_noise, sample_tau, flow_loss
from envs.smb_sequences import SmbSequences, FRAME_SKIP, CONTEXT_FRAMES
from vmodel.train_tokenizer import get_device
import copy
import math
import os
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# The full run on a rented GPU: 19M samples, about 29 passes over the training targets.
# The prototype on the Mac was 30_000 steps at batch 32 with EMA_DECAY 0.999, a bit
# over one pass, and its held-out loss was still falling when the schedule ran out.
# STEPS = 30_000
# BATCH_SIZE = 32
# EMA_DECAY = 0.999
STEPS = 300_000
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WARMUP_STEPS = 1000
EMA_DECAY = 0.9999
CONTEXT_NOISE_MAX = 0.5  # context frames are noised to a random tau in [0, this] during training
STATE_DROPOUT = 0.25  # fraction of samples trained without level id, x_pos and power-up
GRAD_CLIP = 1.0
EVAL_EVERY = 5000
EVAL_BATCHES = 32  # 8 batches of 32 moved the held-out loss by 0.01 from one evaluation to the next
SNAPSHOT_EVERY = 25_000
NUM_WORKERS = 8  # loader processes on a CUDA machine; the Mac loads in the main process
WORLDS = None  # e.g. [7] trains on world 8 only; None is every level
# CHANNELS = (64, 128, 256)  # 19M, the prototype width for the Mac
CHANNELS = (128, 256, 384)  # 46M, the width timed in the browser
CHECKPOINT = "dynamics.pth"
SNAPSHOT_DIR = "data/checkpoints"


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


def save_state(path, props, denoiser, ema, optimizer, scheduler, step):
    # Everything needed to carry on from step, not just the weights: a rented GPU can be
    # taken away mid-run. Written to a second file and renamed, so being stopped during
    # the write leaves the previous checkpoint intact.
    torch.save({
        "props": vars(props),
        "denoiser": denoiser.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step
    }, path + ".tmp")

    os.replace(path + ".tmp", path)


def load_state(path, props, denoiser, ema, optimizer, scheduler, device):
    """Restore a run saved by save_state and return the number of steps it had done."""
    saved = torch.load(path, map_location=device)

    # A checkpoint from another run is never trained over or overwritten: the prototype
    # has no optimizer in it, and a different width cannot be loaded at all.
    if ("optimizer" not in saved or saved["props"] != vars(props)):
        raise RuntimeError(f"{path} is from a different run; move it away before training")

    denoiser.load_state_dict(saved["denoiser"])
    ema.load_state_dict(saved["ema"])
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])

    return saved["step"]


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

    # Half-width floats roughly double the speed on a CUDA card. bfloat16 keeps float32's
    # range, so nothing overflows and no gradient scaler is needed. The loss is taken
    # in float32.
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        velocity = denoiser(
            z_noisy, noisy_context, tau, tau_ctx,
            batch["actions"].to(device), batch["level"].to(device),
            batch["x_pos"].to(device), batch["powerup"].to(device), state_mask
        )

    return flow_loss(velocity.float(), target, eps)


@torch.no_grad()
def evaluate(ema, val_loader, device):
    losses = []

    for i, batch in enumerate(val_loader):
        if (i == EVAL_BATCHES):
            break

        losses.append(compute_loss(ema, batch, device).item())

    return sum(losses) / len(losses)


def batches(loader):
    # One pass over the data is about 10k batches, fewer than STEPS, so the loader is
    # walked again (reshuffled) for as long as the training loop keeps asking.
    while (True):
        for batch in loader:
            yield batch


def main():
    device = get_device()
    print(f"training on {device}")

    if (device.type == "cuda"):
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    train_dset = SmbSequences(train=True, worlds=WORLDS)
    val_dset = SmbSequences(train=False, worlds=WORLDS)
    print(f"{len(train_dset)} training targets  {len(val_dset)} held-out targets")

    # A CUDA card runs faster than one process can cut windows out of the memmap.
    num_workers = NUM_WORKERS if (device.type == "cuda") else 0

    train_loader = DataLoader(
        train_dset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers,
        pin_memory=(device.type == "cuda"), persistent_workers=(num_workers > 0), drop_last=True
    )
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

    first_step = 0

    if (os.path.exists(CHECKPOINT)):
        first_step = load_state(CHECKPOINT, props, denoiser, ema, optimizer, scheduler, device)
        print(f"resuming {CHECKPOINT} from step {first_step}")

    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    running_loss = 0.0
    start = time.time()
    step = first_step - 1  # so a run with nothing left to do saves the step it loaded

    for step, batch in enumerate(batches(train_loader), start=first_step):
        if (step >= STEPS):
            step -= 1
            break

        optimizer.zero_grad()

        loss = compute_loss(denoiser, batch, device)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(denoiser.parameters(), max_norm=GRAD_CLIP)

        optimizer.step()
        scheduler.step()

        # The decay ramps up from 0: at 0.9999 from the start, the average would still
        # be mostly the random initial weights 10k steps in.
        decay = min(EMA_DECAY, (1 + step) / (10 + step))

        with torch.no_grad():
            for ema_param, param in zip(ema.parameters(), denoiser.parameters()):
                ema_param.mul_(decay).add_(param, alpha=1 - decay)

        running_loss += loss.item()

        if (step % 100 == 99):
            print(f"step {step + 1}/{STEPS}  loss {running_loss / 100:.4f}  {time.time() - start:.0f}s", flush=True)
            running_loss = 0.0

        if (step % EVAL_EVERY == EVAL_EVERY - 1):
            print(f"held-out loss (ema) {evaluate(ema, val_loader, device):.4f}", flush=True)

            save_state(CHECKPOINT, props, denoiser, ema, optimizer, scheduler, step + 1)

        # The loss says little about how a rollout looks, so the weights at several
        # points are kept to be played and compared afterwards. load_denoiser reads them.
        if (step % SNAPSHOT_EVERY == SNAPSHOT_EVERY - 1):
            torch.save({"props": vars(props), "ema": ema.state_dict()}, f"{SNAPSHOT_DIR}/dynamics_{step + 1}.pth")

    save_state(CHECKPOINT, props, denoiser, ema, optimizer, scheduler, step + 1)

if (__name__ == "__main__"):
    main()
