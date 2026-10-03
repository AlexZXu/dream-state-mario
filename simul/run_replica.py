from simul.engine import Game, NUM_LEVELS, START_LIVES, RUN_MAX, BRICKS, USED, TILE, TILE_TOP
from simul.decay import Decay
from simul.build_assets import OUT_DIR, TILE_ROWS, WATER
from envs.mario import BUTTONS, FRAME_H, FRAME_W
from envs.smb import NUM_COLORS
import os
import time

import numpy as np
from PIL import Image

SAMPLES = "data/samples/replica.png"
BOT_STEPS = 300  # of each level: 15 seconds at 20 steps per second


def buttons(*names):
    action = np.zeros(len(BUTTONS), dtype=np.int8)

    for name in names:
        action[BUTTONS.index(name)] = 1

    return action


class Bot:
    # Runs right and jumps at whatever is in the way: a wall, a pit, an enemy. It reads
    # the game's own tile grid, which is cheating, but it is here to test the engine.
    def __init__(self):
        self.hold = 0

    def __call__(self, game):
        col = int((game.x + 8) // TILE)
        row = int((game.y + 32 - TILE_TOP) // TILE)

        wall = any(game.solid(col + ahead, row - 1) for ahead in (1, 2))
        pit = not any(game.solid(col + 1, below) for below in range(row, TILE_ROWS)) or not any(game.solid(col + 2, below) for below in range(row, TILE_ROWS))
        enemy = any(0 < other["x"] - game.x < 56 and abs(other["y"] + 8 - game.y) < 40 for other in game.enemies)

        if (game.grounded and self.hold == 0 and (wall or pit or enemy or abs(game.vx) < 0.1)):
            self.hold = 12

        if (game.water and game.y > 120):
            self.hold = 1 if (game.frame % 18 < 3) else 0

        if (self.hold > 0):
            self.hold -= 1
            # A is let go for one step before the next jump, or it is not a new press.
            return buttons("RIGHT", "B", "A") if (self.hold > 0) else buttons("RIGHT", "B")

        return buttons("RIGHT", "B")


def hit_from_below(game, row, col):
    # Puts Mario's head just under a tile, moving up.
    game.x, game.y = float(col * TILE), float(TILE_TOP + (row + 1) * TILE - game.top() + 2)
    game.vx, game.vy = 0.0, -3.0
    game.grounded = False
    game.step(buttons("A"))


def play(game, decay, steps, bot):
    last_scroll = game.scroll
    exact, decayed = [], []

    for _ in range(steps):
        frame = game.step(bot(game))
        dx = max(game.scroll - last_scroll, 0)
        last_scroll = game.scroll

        exact.append(frame)
        decayed.append(decay.step(frame, dx=dx, boxes=game.boxes, cut=game.cut))

    return np.stack(exact), np.stack(decayed)


def main():
    # Every level's assets have the shapes the engine indexes into.
    for level in range(NUM_LEVELS):
        data = np.load(f"{OUT_DIR}/level_{level:02d}.npz")
        H, W = data["background"].shape

        assert H == FRAME_H and W % TILE == 0
        assert data["background"].max() < NUM_COLORS
        assert data["tiles"].shape == (TILE_ROWS, W // TILE)
        assert data["hud"].shape == (TILE_TOP, FRAME_W)
        assert len(data["pieces"]) >= 1 and data["pieces"][0][0] == 0
        assert data["lift_lengths"].sum() == len(data["lift_paths"])

    print(f"{NUM_LEVELS} levels load")

    # Mario lands on the ground at the start of every level and stays in play.
    noop = buttons()
    standing = 0

    for level in range(NUM_LEVELS):
        game = Game(level)

        for _ in range(40):
            frame = game.step(noop)

        assert frame.shape == (FRAME_H, FRAME_W) and frame.dtype == np.uint8 and frame.max() < NUM_COLORS
        assert game.mode == "play", f"level {level} kills Mario at the start"
        standing += int(game.grounded)

        if (game.area_type != WATER):
            assert game.grounded, f"level {level} starts in the air"

    print(f"Mario is on the ground a moment after the start of {standing} / {NUM_LEVELS} levels")

    # Movement matches the original game's numbers: top running speed, and a standing
    # jump of about four blocks.
    game = Game(0)
    for _ in range(40):
        game.step(noop)

    ground = game.y
    highest = ground

    for step in range(40):
        game.step(buttons("A") if (step < 20) else noop)
        highest = min(highest, game.y)

    assert game.grounded and game.y == ground
    assert 3.5 * TILE <= ground - highest <= 5 * TILE, ground - highest

    start_x = game.x
    for _ in range(40):
        game.step(buttons("RIGHT", "B"))

    assert abs(game.vx - RUN_MAX) < 0.1
    assert game.x > start_x + 150 and game.scroll > 0
    print(f"standing jump {ground - highest:.0f} px high; running speed {game.vx:.2f} px per frame")

    # The rules, one at a time, by putting Mario where each one applies. Dropping onto
    # an enemy is a stomp.
    game = Game(0)
    while (len(game.enemies) == 0):
        game.step(buttons("RIGHT"))

    enemy = game.enemies[0]
    game.x, game.y, game.vx, game.vy = enemy["x"], enemy["y"] - 40, 0.0, 1.0
    game.grounded = False

    for _ in range(6):
        game.step(noop)

    assert game.mode == "play" and game.score == 100 and game.vy != 0
    assert len(game.enemies) == 0 or game.enemies[0]["squashed"] > 0

    # Walking into one is a death, then a respawn at the start with a life gone.
    game = Game(0)
    steps = 0

    while (game.mode == "play" and steps < 200):
        game.step(buttons("RIGHT"))
        steps += 1

    assert game.mode == "dying"

    while (game.mode == "dying"):
        game.step(noop)

    assert game.cut and game.lives == START_LIVES - 1 and game.level == 0 and game.x == game.pieces[0][2]

    # Hitting a question block from below pays a coin and uses the block up; the one
    # with the mushroom makes Mario big, and big Mario breaks bricks.
    row, col = [int(v[0]) for v in np.nonzero(game.tiles == 0xC0)]
    hit_from_below(game, row, col)

    assert game.coins == 1 and game.tiles[row, col] == USED

    row, col = [int(v[0]) for v in np.nonzero(game.tiles == 0xC1)]
    hit_from_below(game, row, col)

    assert game.tiles[row, col] == USED and len(game.items) == 1
    game.x, game.y = game.items[0]["x"], game.items[0]["y"] - 8
    game.step(noop)
    assert game.big and len(game.items) == 0

    row, col = [int(v[0]) for v in np.nonzero(game.tiles == BRICKS[0])]
    hit_from_below(game, row, col)
    assert game.tiles[row, col] == 0

    # Reaching the end of what was recorded cuts to the next piece of the level, and
    # from the last piece to the next level. In 1-1 the first cut is the bonus-room pipe.
    game = Game(0)
    assert len(game.pieces) == 2

    game.x, game.vx = float(game.pieces[0][4] - 4), 1.5
    game.step(buttons("RIGHT"))
    assert game.piece == 1 and game.cut and game.scroll >= game.pieces[1][0]

    steps = 0
    while (game.level == 0 and steps < 600):
        game.step(buttons("RIGHT", "B", "A") if (steps % 14 < 12) else buttons("RIGHT", "B"))
        steps += 1

    assert game.level == 1 and game.cut, "the flagpole in 1-1 does not lead to 1-2"
    print("stomp, death and respawn, blocks, mushroom, the cut across the pipe and the flagpole all work")

    # The game has no randomness of its own: the same buttons give the same frames.
    first, _ = play(Game(0), Decay(), 300, Bot())
    second, _ = play(Game(0), Decay(), 300, Bot())
    assert np.array_equal(first, second)

    # A bot plays the opening of every level. It is not good at the game; the point is
    # that nothing breaks anywhere: platforms, water, castles, deaths, respawns.
    start = time.time()
    exact, decayed, progress = [], [], []

    for level in range(NUM_LEVELS):
        game = Game(level)
        frames, filtered = play(game, Decay(), BOT_STEPS, Bot())
        exact.append(frames)
        decayed.append(filtered)
        progress.append(frames.shape[0])

    exact, decayed = np.concatenate(exact), np.concatenate(decayed)
    rate = len(exact) / (time.time() - start)

    assert decayed.shape == exact.shape and decayed.dtype == np.uint8 and decayed.max() < NUM_COLORS
    assert rate > 60, f"{rate:.0f} steps per second is too slow for 20 Hz with room to spare"
    print(f"bot played {BOT_STEPS} steps of each level at {rate:.0f} steps per second, engine and decay together")

    # The decay is visible but leaves the game readable: a "pretty good" world model
    # gets a few percent of pixels wrong, not a fifth of them.
    wrong = (decayed != exact).mean(axis=(1, 2))
    assert 0.002 < wrong.mean() < 0.06, wrong.mean()
    assert wrong.max() < 0.25, wrong.max()
    print(f"decay changes {wrong.mean():.2%} of pixels on average, {wrong.max():.2%} at worst")

    # Strength 0 is the exact game, so D in the player is a true before/after.
    _, untouched = play(Game(0), Decay(strength=0.0), 200, Bot())
    again, _ = play(Game(0), Decay(), 200, Bot())
    assert np.array_equal(untouched, again)
    palette = game.sprites.palette

    # A sheet to look at: exact on top, decayed below, from eight of the levels.
    picks = np.arange(120, len(exact), 4 * BOT_STEPS)[:8]
    sheet = np.concatenate([np.concatenate(list(exact[picks]), axis=1), np.concatenate(list(decayed[picks]), axis=1)], axis=0)

    os.makedirs(os.path.dirname(SAMPLES), exist_ok=True)
    Image.fromarray(palette[sheet]).save(SAMPLES)
    print(f"wrote {SAMPLES}")

    print("\nAll tests passed.")

if (__name__ == "__main__"):
    main()
