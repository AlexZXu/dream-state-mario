from envs.mario import make_env, step, clean, save_state, load_state, BUTTONS, FRAME_H, FRAME_W
import os
import time

import numpy as np
import pygame

SCALE = 3  # window size as a multiple of the 256x224 frame
STEP_HZ = 20
SNAPSHOT_EVERY = 40  # steps between savestates: one every 2 seconds
STATE_DIR = "data/states"
OUT_DIR = "data/human"

KEYS = {
    pygame.K_LEFT: "LEFT",
    pygame.K_RIGHT: "RIGHT",
    pygame.K_UP: "UP",
    pygame.K_DOWN: "DOWN",
    pygame.K_x: "A",  # jump
    pygame.K_z: "B"   # run / fireball
}


def read_keys():
    pressed = pygame.key.get_pressed()
    action = np.zeros(len(BUTTONS), dtype=np.int8)

    for key, name in KEYS.items():
        action[BUTTONS.index(name)] = pressed[key]

    return clean(action)


def flush(session, segment, frames, actions, rams):
    if (len(frames) == 0):
        return

    path = f"{OUT_DIR}/session_{session}_{segment:03d}.npz"
    np.savez_compressed(path, frames=np.stack(frames), actions=np.stack(actions), ram=np.stack(rams))
    print(f"saved {path}  {len(frames)} steps")


def main():
    os.makedirs(STATE_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    pygame.init()
    screen = pygame.display.set_mode((FRAME_W * SCALE, FRAME_H * SCALE))
    pygame.display.set_caption("arrows move, X jump, Z run, backspace rewind, R restart, esc quit")
    clock = pygame.time.Clock()

    env = make_env()
    env.reset()

    session = int(time.time())
    segment = 0
    snapshots = []

    frames = []
    actions = []
    rams = []

    running = True

    while (running):
        rewind = False
        restart = False

        for event in pygame.event.get():
            if (event.type == pygame.QUIT):
                running = False

            if (event.type == pygame.KEYDOWN):
                if (event.key == pygame.K_ESCAPE):
                    running = False
                if (event.key == pygame.K_BACKSPACE):
                    rewind = True
                if (event.key == pygame.K_r):
                    restart = True

        # A rewind or restart is a jump cut. Each saved file has to be one unbroken stretch
        # of play, because training windows are cut from consecutive steps, so what was
        # recorded so far is written out and a new segment starts.
        if (rewind or restart):
            flush(session, segment, frames, actions, rams)
            segment += 1
            frames, actions, rams = [], [], []

            if (rewind and len(snapshots) > 0):
                # The newest snapshot can be a moment before the mistake, so go back two.
                if (len(snapshots) > 1):
                    snapshots.pop()

                load_state(env, snapshots[-1])
            else:
                env.reset()

        # The keyboard is read once per step and held for the whole step, the same as the
        # website will do, so the recorded control has the timing the model will be given.
        action = read_keys()
        frame, ram, done = step(env, action)

        frames.append(frame)
        actions.append(action)
        rams.append(ram)

        if (len(frames) % SNAPSHOT_EVERY == 0):
            path = f"{STATE_DIR}/session_{session}_{segment:03d}_{len(frames):06d}.state"
            save_state(env, path)
            snapshots.append(path)

        if (done):
            flush(session, segment, frames, actions, rams)
            segment += 1
            frames, actions, rams = [], [], []
            env.reset()

        # frame shape: (224, 256, 3) -- pygame surfaces index (x, y), so the axes swap.
        surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        screen.blit(pygame.transform.scale(surface, screen.get_size()), (0, 0))
        pygame.display.flip()

        clock.tick(STEP_HZ)

    flush(session, segment, frames, actions, rams)

    env.close()
    pygame.quit()

if (__name__ == "__main__"):
    main()
