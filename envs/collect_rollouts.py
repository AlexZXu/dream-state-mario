from envs.mario import make_env, rollout, load_state, decode, NoisyPolicy, PLAYING
import glob
import os
import time
from multiprocessing import Pool

import numpy as np

NUM_ROLLOUTS = 750
ROLLOUT_STEPS = 400  # 20 seconds at 20 steps per second
NUM_WORKERS = 8
STATE_DIR = "data/states"
OUT_DIR = "data/rollouts"


def playable_states(env):
    # The recorder snapshots on a timer, so some savestates land in the middle of a death
    # or on the black screen between lives. A rollout started there would spend its first
    # seconds on nothing, so only states with Mario under control are kept.
    paths = []

    for path in sorted(glob.glob(f"{STATE_DIR}/*.state")):
        load_state(env, path)

        if (decode(env.get_ram())["player_state"] == PLAYING):
            paths.append(path)

    return paths


def collect(worker):
    # The emulator allows one instance per process, so each worker owns its own.
    env = make_env()
    env.reset()

    paths = playable_states(env)

    if (worker == 0):
        print(f"{len(paths)} playable savestates in {STATE_DIR}")

    start = time.time()

    for i in range(worker, NUM_ROLLOUTS, NUM_WORKERS):
        # Seeded by the rollout index, so the dataset is the same whichever worker runs it.
        rng = np.random.default_rng(i)

        # A noisy policy rarely survives past the first few screens. Starting from
        # savestates spread along a human playthrough is what puts the far end of the
        # level in the data without training an agent to get there.
        if (len(paths) > 0):
            load_state(env, paths[rng.integers(len(paths))])
        else:
            env.reset()

        frames, actions, rams = rollout(env, ROLLOUT_STEPS, NoisyPolicy(rng))

        np.savez_compressed(f"{OUT_DIR}/rollout_{i:05d}.npz", frames=frames, actions=actions, ram=rams)

        if (worker == 0 and (i // NUM_WORKERS) % 10 == 0):
            elapsed = time.time() - start
            print(f"rollout {i}/{NUM_ROLLOUTS}  frames {frames.shape}  actions {actions.shape}  {elapsed:.0f}s")

    env.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    start = time.time()

    with Pool(processes=NUM_WORKERS) as pool:
        pool.map(collect, range(NUM_WORKERS))

    print(f"done  {NUM_ROLLOUTS} rollouts in {time.time() - start:.0f}s")

if (__name__ == "__main__"):
    main()
