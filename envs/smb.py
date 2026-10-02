from envs.mario import BUTTONS, CROP_TOP, FRAME_H, FRAME_W
import glob
import io
import os
import re

import numpy as np
from PIL import Image

# Super Mario Bros. Gameplay Dataset, R.C. Pinto 2021, CC-BY-4.0
# https://github.com/rafaelcp/smbdataset
URL = "https://media.githubusercontent.com/media/rafaelcp/smbdataset/main/data-smb.7z"

DATA_DIR = "data/smb"
ARCHIVE = f"{DATA_DIR}/data-smb.7z"
RAW_DIR = f"{DATA_DIR}/raw"

NUM_COLORS = 64  # the NES master palette
PALETTE_OFFSET = 128  # the recorder's PNG palette keeps the 64 NES colours at entries 128..191
RAM_SIZE = 2048
FPS = 60  # every NES frame was recorded, not every third one

# The dataset packs the pad into one byte, most significant bit first:
# A, up, left, B, start, right, down, select. Start and select are never pressed in it.
BUTTON_BITS = {"A": 128, "UP": 64, "LEFT": 32, "B": 16, "RIGHT": 4, "DOWN": 2}

# <user>_<session>_e<episode>_<world>-<level>_f<frame>_a<action>_<datetime>.<outcome>.png
NAME = re.compile(r"(.+)_(.+)_e(\d+)_(\d)-(\d)_f(\d+)_a(\d+)_(.+)\.(win|fail)\.png")

EPISODE_DTYPE = np.dtype([
    ("start", np.int64),   # index of the episode's first frame in the big arrays
    ("length", np.int64),
    ("world", np.int8),    # 0-indexed, the way the game's RAM counts them
    ("stage", np.int8),
    ("win", np.bool_)      # reached the end of the level, as opposed to dying
])


def to_action(byte):
    # one byte -> (6,) in BUTTONS order
    return np.array([(byte & BUTTON_BITS[name]) > 0 for name in BUTTONS], dtype=np.int8)


def read_ram(raw):
    # The recorder appended its own chunks after the end of the PNG. Each is a label, a
    # zero byte, the data, and then 8 bytes of length and checksum before the next label.
    end_of_image = raw.find(b"IEND")
    start = raw.find(b"tEXtRAM", end_of_image) + len(b"tEXtRAM") + 1
    end = raw.find(b"tEXtBP1", end_of_image) - 8

    # The recorder wrote the RAM in text mode on Windows, which turned every byte 13 into
    # the pair 13, 10. Collapsing the pair back is exact: a 13 that was really followed
    # by a 10 came out as 13, 10, 10 and collapses to 13, 10 again.
    # https://github.com/rafaelcp/smbdataset/issues/4
    ram = raw[start:end].replace(b"\r\n", b"\r")

    return np.frombuffer(ram, dtype=np.uint8)


def read_frame(path):
    with open(path, "rb") as f:
        raw = f.read()

    image = Image.open(io.BytesIO(raw))

    # pixels shape: (240, 256) uint8 -- palette entries, not colours
    pixels = np.asarray(image)
    palette = np.array(image.getpalette(), dtype=np.uint8).reshape(-1, 3)
    palette = palette[PALETTE_OFFSET:PALETTE_OFFSET + NUM_COLORS]

    # Subtracting the offset leaves the NES colour number itself. An entry below the
    # offset would wrap around past 64 in uint8, so the one check covers both directions.
    frame = pixels[CROP_TOP:CROP_TOP + FRAME_H] - PALETTE_OFFSET
    assert frame.shape == (FRAME_H, FRAME_W)
    assert frame.max() < NUM_COLORS

    action = to_action(int(NAME.match(os.path.basename(path)).group(7)))

    return frame, action, read_ram(raw), palette


def scan():
    # One entry per episode: (folder name, world, stage, win, frame paths in play order).
    episodes = []

    for folder in sorted(glob.glob(f"{RAW_DIR}/*/")):
        names = [name for name in os.listdir(folder) if (name.endswith(".png"))]
        # f9 sorts after f1000 as text, so the frame number has to be compared as a number.
        names.sort(key=lambda name: int(NAME.match(name).group(6)))

        match = NAME.match(names[0])
        world = int(match.group(4)) - 1
        stage = int(match.group(5)) - 1
        win = (match.group(9) == "win")

        paths = [os.path.join(folder, name) for name in names]
        episodes.append((os.path.basename(os.path.dirname(folder)), world, stage, win, paths))

    return episodes


def load():
    # frames is 42 GB, so everything is memory-mapped and only the frames a batch touches
    # are ever read from disk.
    return {
        "frames": np.load(f"{DATA_DIR}/frames.npy", mmap_mode="r"),    # (N, 224, 256) uint8, NES colour numbers
        "actions": np.load(f"{DATA_DIR}/actions.npy", mmap_mode="r"),  # (N, 6) int8, held during that frame
        "ram": np.load(f"{DATA_DIR}/ram.npy", mmap_mode="r"),          # (N, 2048) uint8
        "episodes": np.load(f"{DATA_DIR}/episodes.npy"),               # (280,) EPISODE_DTYPE
        "palette": np.load(f"{DATA_DIR}/palette.npy")                  # (64, 3) uint8, colour number -> RGB
    }
