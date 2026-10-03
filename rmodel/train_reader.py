from rmodel.reader import Reader, ReaderProps, reader_loss
from envs.smb_events import SmbEvents, MAX_DX
from vmodel.train_tokenizer import get_device
from mmodel.train_dynamics import batches
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

STEPS = 20_000
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EVAL_EVERY = 5000
EVAL_BATCHES = 40
CHECKPOINT = "reader.pth"


def precision_recall(logits, target):
    predicted = (logits > 0)
    target = (target > 0.5)
    hits = (predicted & target).sum().item()

    return hits / max(predicted.sum().item(), 1), hits / max(target.sum().item(), 1)


@torch.no_grad()
def evaluate(reader, val_loader, device):
    reader.eval()

    outputs = {"dx": [], "died": [], "clear": [], "powerup": []}
    targets = {"dx": [], "dx_valid": [], "died": [], "clear": [], "powerup": []}

    for i, batch in enumerate(val_loader):
        if (i == EVAL_BATCHES):
            break

        output = reader(batch["pair"].to(device))

        for name in outputs:
            outputs[name].append(output[name].cpu())

        for name in targets:
            targets[name].append(batch[name])

    outputs = {name: torch.cat(values) for name, values in outputs.items()}
    targets = {name: torch.cat(values) for name, values in targets.items()}

    reader.train()

    valid = (targets["dx_valid"] > 0.5)
    dx_error = (outputs["dx"] - targets["dx"]).abs()[valid].mean().item() * MAX_DX

    return {
        "dx_error": dx_error,  # mean absolute error in pixels per step
        "died": precision_recall(outputs["died"], targets["died"]),
        "clear": precision_recall(outputs["clear"], targets["clear"]),
        "powerup": (outputs["powerup"].argmax(dim=1) == targets["powerup"]).float().mean().item()
    }


def main():
    device = get_device()
    print(f"training on {device}")

    train_dset = SmbEvents(train=True)
    val_dset = SmbEvents(train=False)
    print(f"{len(train_dset)} training pairs  {len(val_dset)} held-out pairs")

    train_loader = DataLoader(train_dset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    reader = Reader(ReaderProps()).to(device)
    print(f"reader params {sum(param.numel() for param in reader.parameters()) / 1e6:.2f}M")

    optimizer = optim.Adam(params=reader.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS)

    running_loss = 0.0
    start = time.time()

    for step, batch in enumerate(batches(train_loader)):
        if (step == STEPS):
            break

        batch = {name: value.to(device) for name, value in batch.items()}

        optimizer.zero_grad()

        loss = reader_loss(reader(batch["pair"]), batch)
        loss.backward()

        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        if (step % 100 == 99):
            print(f"step {step + 1}/{STEPS}  loss {running_loss / 100:.4f}  {time.time() - start:.0f}s")
            running_loss = 0.0

        if (step % EVAL_EVERY == EVAL_EVERY - 1):
            scores = evaluate(reader, val_loader, device)
            print(
                f"held-out: dx off by {scores['dx_error']:.2f} px per step  "
                f"died precision {scores['died'][0]:.3f} recall {scores['died'][1]:.3f}  "
                f"clear precision {scores['clear'][0]:.3f} recall {scores['clear'][1]:.3f}  "
                f"powerup {scores['powerup'] * 100:.1f}%"
            )

            torch.save(reader.state_dict(), CHECKPOINT)

    torch.save(reader.state_dict(), CHECKPOINT)

if (__name__ == "__main__"):
    main()
