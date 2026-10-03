from simul.engine import Game, NUM_LEVELS
from mmodel.dreamer import Dreamer
from vmodel.train_tokenizer import get_device
from envs.play_record import read_keys, SCALE, STEP_HZ
from envs.mario import FRAME_H, FRAME_W
import time

import numpy as np
import pygame

TAU_GUIDE = 0.6  # how much of the simulation's frame is replaced by noise before the model redraws it
TAU_STEP = 0.1


STILL_TAU = 0.4  # the most noise used while Mario stands still or has barely started moving
SLOW = 1.0  # pixels per NES frame under which he counts as that; walking pace is 1.56


class Redream:
    """The simulation keeps the game's state and the world model draws every frame.
    Each step the simulation's exact frame is noised and the model generates from there,
    so what is on screen is the model's output, with its own errors, but it cannot lose
    Mario or wander into another level the way it does on its own."""

    def __init__(self, game, dreamer, tau_guide=TAU_GUIDE, anchor=False):
        self.game = game
        self.dreamer = dreamer
        self.tau_guide = tau_guide
        self.anchor = anchor

        dreamer.use_state(True)
        self.reseed(game.render())

    def powerup(self):
        return 2 if (self.game.fire) else int(self.game.big)

    def reseed(self, frame):
        self.dreamer.seed_frames(frame, self.game.level, self.game.x, self.powerup())
        self.waiting = True

    def step(self, action):
        # Returns the simulation's exact frame and the model's version of it.
        game = self.game
        frame = game.step(action)

        # A respawn or a new level is a new scene: the model is started again from it,
        # the same as play.dream starting from a seed clip.
        if (game.cut):
            self.reseed(frame)

        # Until the player first touches a button in a scene, the model is not used at
        # all. The prototype erases a Mario who stands still (the recordings almost
        # never show it: run is held 83% of the time), and the opening of a level is
        # exactly that. The scene goes on moving meanwhile, so the model's memory is
        # kept on the latest frame.
        if (self.waiting and not np.any(action)):
            self.dreamer.seed_frames(frame, game.level, game.x, self.powerup())
            self.waiting = True

            return frame, frame

        self.waiting = False

        # The same weakness later on, whenever Mario stops, and for the first steps of
        # setting off again: for those the model is given the simulation's past frames
        # instead of its own, and less noise, so it has no chance to forget him.
        still = (game.mode == "play" and abs(game.vx) < SLOW)
        tau_guide = min(self.tau_guide, STILL_TAU) if (still) else self.tau_guide

        self.dreamer.set_state(game.level, game.x, self.powerup())
        dreamed = self.dreamer.step(action, guide=frame, tau_guide=tau_guide, anchor=(self.anchor or still))

        return frame, dreamed


def main():
    game = Game()
    redream = Redream(game, Dreamer(get_device()))
    dreamer = redream.dreamer

    pygame.init()
    screen = pygame.display.set_mode((FRAME_W * SCALE, FRAME_H * SCALE))
    pygame.display.set_caption("redream: arrows move, X jump, Z run, [ ] less/more model, H true history, D exact, N next level, R restart, esc quit")
    clock = pygame.time.Clock()

    exact = False

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
                    game.load(game.level)
                if (event.key == pygame.K_n):
                    game.load((game.level + 1) % NUM_LEVELS)
                if (event.key == pygame.K_d):
                    exact = not exact
                    print("showing the simulation's exact frame" if (exact) else "showing the model's frame")
                if (event.key == pygame.K_h):
                    # With true history the model is given the simulation's past frames
                    # instead of its own, so each frame is a one-step prediction.
                    redream.anchor = not redream.anchor
                    print(f"true history {'on' if (redream.anchor) else 'off'}")
                if (event.key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET)):
                    change = TAU_STEP if (event.key == pygame.K_RIGHTBRACKET) else -TAU_STEP
                    redream.tau_guide = float(np.clip(redream.tau_guide + change, 0, 1))
                    print(f"guide noise {redream.tau_guide:.1f}  (0 is the simulation, 1 is the model alone)")

        frame, dreamed = redream.step(read_keys())

        frames += 1
        shown = frame if (exact) else dreamed

        surface = pygame.surfarray.make_surface(dreamer.rgb(shown).swapaxes(0, 1))
        screen.blit(pygame.transform.scale(surface, screen.get_size()), (0, 0))
        pygame.display.flip()

        if (frames % 100 == 0):
            print(f"{frames / (time.time() - start):.1f} frames per second")

        # Capped at the game's rate; a slow model just runs the game in slow motion.
        clock.tick(STEP_HZ)

    pygame.quit()

if (__name__ == "__main__"):
    main()
