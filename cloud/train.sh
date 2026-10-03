#!/bin/bash
# Step 5 on a rented CUDA machine: build the dataset, encode it, start the full dynamics run.
#
#   git clone https://github.com/AlexZXu/dream-state-mario.git && cd dream-state-mario
#   (copy tokenizer.pth from the Mac into this folder)
#   bash cloud/train.sh
#
# Needs Python 3.12 and about 80 GB of disk: the archive and its PNGs are 6.5 GB, frames
# 42 GB, RAM 1.5 GB, latents 10.6 GB, and the checkpoints about 2.5 GB.
#
# Safe to run again after the machine is stopped: each stage is skipped once its last
# file exists, and train_dynamics resumes from dynamics.pth.
set -e

if [ ! -f tokenizer.pth ]; then
    echo "tokenizer.pth is missing; copy it from the Mac first"
    exit 1
fi

if [ ! -d .venv ]; then
    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
fi

mkdir -p data/logs

# episodes.npy is the last thing convert_smb writes.
if [ ! -f data/smb/episodes.npy ]; then
    .venv/bin/python -u -m envs.convert_smb
fi

# The latents must come from this exact tokenizer.pth. latent_stats.npy is written last.
if [ ! -f data/smb/latent_stats.npy ]; then
    .venv/bin/python -u -m vmodel.encode_dataset
fi

.venv/bin/python -m mmodel.run_dynamics

nohup .venv/bin/python -u -m mmodel.train_dynamics >> data/logs/train_dynamics.log 2>&1 &

echo "training started; follow it with: tail -f data/logs/train_dynamics.log"
