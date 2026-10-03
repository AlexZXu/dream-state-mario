from simul.engine import Game, NUM_LEVELS
from simul.decay import Decay
from envs.play_record import read_keys, SCALE, STEP_HZ
from envs.mario import FRAME_H, FRAME_W
import numpy as np
import pygame

STRENGTH_STEP = 0.25


def main():
    game = Game()
    decay = Decay(seed=None)

    pygame.init()
    screen = pygame.display.set_mode((FRAME_W * SCALE, FRAME_H * SCALE))
    pygame.display.set_caption("replica: arrows move, X jump, Z run, R restart, N next level, D decay on/off, [ ] decay amount, esc quit")
    clock = pygame.time.Clock()

    decaying = True
    last_scroll = game.scroll
    running = True

    while (running):
        for event in pygame.event.get():
            if (event.type == pygame.QUIT):
                running = False

            if (event.type == pygame.KEYDOWN):
                if (event.key == pygame.K_ESCAPE):
                    running = False
                if (event.key == pygame.K_r):
                    game.load(game.level)
                if (event.key == pygame.K_n):
                    game.load((game.level + 1) % NUM_LEVELS)
                if (event.key == pygame.K_d):
                    decaying = not decaying
                    print(f"decay {'on' if (decaying) else 'off: this is the exact frame the engine draws'}")
                if (event.key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET)):
                    change = STRENGTH_STEP if (event.key == pygame.K_RIGHTBRACKET) else -STRENGTH_STEP
                    decay.strength = float(np.clip(decay.strength + change, 0, 3))
                    print(f"decay strength {decay.strength:.2f}")

        # One step is 3 NES frames, shown at 20 Hz: the rate the world model runs at.
        frame = game.step(read_keys())

        # The filter always runs, so its memory stays in step with the game even while
        # the exact frame is the one on screen.
        dx = max(game.scroll - last_scroll, 0)
        last_scroll = game.scroll
        decayed = decay.step(frame, dx=dx, boxes=game.boxes, cut=game.cut)

        if (decaying):
            frame = decayed

        # frame shape: (224, 256) colour numbers -> (224, 256, 3); pygame indexes (x, y).
        surface = pygame.surfarray.make_surface(game.sprites.palette[frame].swapaxes(0, 1))
        screen.blit(pygame.transform.scale(surface, screen.get_size()), (0, 0))
        pygame.display.flip()

        clock.tick(STEP_HZ)

    pygame.quit()

if (__name__ == "__main__"):
    main()
