from envs.mario import BUTTONS, FRAME_H, FRAME_W
from envs.smb import (
    scan, read_frame, URL, DATA_DIR, ARCHIVE, RAW_DIR, RAM_SIZE, EPISODE_DTYPE
)
import os
import time
import urllib.request
from multiprocessing import Pool

import numpy as np
import py7zr

NUM_WORKERS = 8


def download():
    if (os.path.exists(ARCHIVE)):
        return

    print(f"downloading {URL} (890 MB)")
    urllib.request.urlretrieve(URL, ARCHIVE)


def extract():
    if (os.path.exists(RAW_DIR)):
        return

    print(f"extracting {ARCHIVE} (737k files, about 5 minutes)")

    with py7zr.SevenZipFile(ARCHIVE) as archive:
        archive.extractall(path=RAW_DIR)


def convert(job):
    start, paths = job

    # Every worker maps the same files and writes only its own episode's rows, so nothing
    # is passed back through the pool except the palette and a count.
    frames = np.load(f"{DATA_DIR}/frames.npy", mmap_mode="r+")
    actions = np.load(f"{DATA_DIR}/actions.npy", mmap_mode="r+")
    rams = np.load(f"{DATA_DIR}/ram.npy", mmap_mode="r+")

    palette = None
    bad_ram = 0

    for i, path in enumerate(paths):
        frame, action, ram, frame_palette = read_frame(path)

        # The pixels are stored as colour numbers, which only works if every file agrees
        # on what each number means.
        if (palette is None):
            palette = frame_palette

        assert np.array_equal(palette, frame_palette)

        frames[start + i] = frame
        actions[start + i] = action

        if (len(ram) == RAM_SIZE):
            rams[start + i] = ram
        else:
            bad_ram += 1

    frames.flush()
    actions.flush()
    rams.flush()

    return palette, bad_ram


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    download()
    extract()

    scanned = scan()
    episodes = np.zeros(len(scanned), dtype=EPISODE_DTYPE)
    jobs = []
    N = 0

    for i, (name, world, stage, win, paths) in enumerate(scanned):
        episodes[i] = (N, len(paths), world, stage, win)
        jobs.append((N, paths))
        N += len(paths)

    print(f"{len(scanned)} episodes  {N} frames  {N / 60 / 60:.0f} minutes of play")

    # open_memmap writes the .npy header and reserves the space without holding the
    # array in memory; frames alone is N * 224 * 256 bytes.
    np.lib.format.open_memmap(f"{DATA_DIR}/frames.npy", mode="w+", dtype=np.uint8, shape=(N, FRAME_H, FRAME_W))
    np.lib.format.open_memmap(f"{DATA_DIR}/actions.npy", mode="w+", dtype=np.int8, shape=(N, len(BUTTONS)))
    np.lib.format.open_memmap(f"{DATA_DIR}/ram.npy", mode="w+", dtype=np.uint8, shape=(N, RAM_SIZE))

    start = time.time()

    with Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(convert, jobs, chunksize=1)

    palette = results[0][0]
    bad_ram = sum(count for _, count in results)

    for other, _ in results:
        assert np.array_equal(palette, other)

    np.save(f"{DATA_DIR}/episodes.npy", episodes)
    np.save(f"{DATA_DIR}/palette.npy", palette)

    print(f"done in {time.time() - start:.0f}s  frames with unreadable RAM: {bad_ram}")

if (__name__ == "__main__"):
    main()
