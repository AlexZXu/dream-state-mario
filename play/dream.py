from mmodel.dreamer import Dreamer
from vmodel.train_tokenizer import get_device
from envs.play_record import read_keys, SCALE, STEP_HZ
from envs.mario import FRAME_H, FRAME_W
import time

import numpy as np
import pygame


def main():
    device = get_device()
    dreamer = Dreamer(device)

    pygame.init()
    screen = pygame.display.set_mode((FRAME_W * SCALE, FRAME_H * SCALE))
    pygame.display.set_caption("dream: arrows move, X jump, Z run, R new start, S toggle state, esc quit")
    clock = pygame.time.Clock()

    rng = np.random.default_rng()
    use_state = False

    dreamer.seed(rng.integers(len(dreamer.dset)))

    running = True
    frames = 0
    start = time.time()

    while (running):
        for event in pygame.event.get():
            if (event.type == pygame.QUIT):
                running = False

            if (event.type == pygame.KEYDOWN):
                if (event.key == pygame.K_ESCAPE):
                    running = False
                if (event.key == pygame.K_r):
                    dreamer.seed(rng.integers(len(dreamer.dset)))
                if (event.key == pygame.K_s):
                    use_state = not use_state
                    dreamer.use_state(use_state)
                    print(f"state conditioning {'on' if (use_state) else 'off'}")

        frame = dreamer.step(read_keys())
        frames += 1

        surface = pygame.surfarray.make_surface(dreamer.rgb(frame).swapaxes(0, 1))
        screen.blit(pygame.transform.scale(surface, screen.get_size()), (0, 0))
        pygame.display.flip()

        if (frames % 100 == 0):
            print(f"{frames / (time.time() - start):.1f} frames per second")

        # Capped at the game's rate; a slow model just runs the game in slow motion.
        clock.tick(STEP_HZ)

    pygame.quit()

if (__name__ == "__main__"):
    main()
