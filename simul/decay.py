from envs.mario import FRAME_H, FRAME_W
from simul.build_assets import OUT_DIR, DIGIT_Y
import numpy as np

CELL = 8  # the tokenizer's latent grid is 28x32, one cell per 8x8 pixels
ROWS = FRAME_H // CELL
COLS = FRAME_W // CELL

COOL = 0.82  # how much of a cell's uncertainty is left a step later
BASE_HEAT = 0.004  # every cell is a little unsure all the time
EDGE_HEAT = 0.6  # scenery that has just scrolled in was never seen before
MOTION_HEAT = 0.07  # added each step to the cells under anything that moves
HUD_HEAT = 0.02  # the HUD letters stay crisp in the real model; it is the digits that go wrong
FLARE_CHANCE = 0.04  # per step, that some patch of the screen loses its footing
FLARE_HEAT = 0.6
SPREAD = 0.35  # how much uncertainty leaks into neighbouring cells each step
DRIFT_STEP = 0.001  # the whole picture's uncertainty wanders up and down slowly
DRIFT_MAX = 0.008

JITTER = 0.9  # chance that a pixel in a fully uncertain cell is copied from a neighbour
LAG = 0.45  # chance that it keeps what was there in the last frame instead
RESAMPLE = 0.25  # share of the random choices redrawn each step; the rest persist
PATCH = 4  # neighbour offsets are shared by 4x4 patches, so edges wobble instead of turning to sand
OFFSETS = np.array([-2, -1, -1, 0, 0, 1, 1, 2], dtype=np.int64)

GUESS_CHANCE = 0.18  # that a cell scrolling in is first drawn as scenery from further left
GUESS_HEAL = 0.4  # chance per step that such a guess is corrected
HUD_ROWS = slice(1, 3)

# What the prototype (play.dream) does over a rollout, copied from looking at its frames:
# small things shrink and vanish, the texture inside big things melts into its main
# colour in patches, and the HUD digits stay sharp but stop being the right digits.
ROT_CHANCE = 0.002  # per step, that a cell with something in it starts to go
ROT_SPREAD = 0.006  # per step, that rot takes a neighbouring cell with it
ROT_STOP = 0.01  # per step, that a rotting cell stops and starts to come back
FADE_RATE = 0.02  # a cell is fully gone 50 steps (2.5 seconds) after it starts
FADE_BACK = 0.01
MELT_MAX = 0.6  # the most of a big structure's texture that melts; small things go completely
ALONE = 0.6  # share of backdrop around a cell above which it is a small thing on its own
GRAIN = 2  # pixels vanish in 2x2 chunks
NEAR_MARIO = (4, 3)  # cells (rows, cols) around Mario kept whole, so the game stays playable

DIGIT_STALL = 0.45  # chance that a digit which should change keeps showing the old one
DIGIT_MUTATE = 0.0015  # per step, that a correct digit turns into a similar-looking one
DIGIT_HEAL = 0.01  # per step, that a wrong digit is put right
DIGIT_HYBRID = 0.3  # share of wrong digits that are half one digit, half another
LOOKALIKES = {0: (8, 6, 9), 1: (7, 4), 2: (7, 3), 3: (8, 9, 5), 4: (1, 9), 5: (6, 3, 8), 6: (8, 5, 0), 7: (1, 2), 8: (0, 3, 6, 9), 9: (8, 3, 0)}


class Decay:
    """Makes exact frames look generated. A world model with a per-pixel colour head
    (like vmodel's decoder) does not blur: flat areas stay perfect and the errors are
    wrong-but-plausible colours along edges, worst where things move or have just come
    into view, and they persist because each frame is drawn from the last ones. This
    imitates that with an uncertainty map on the latent grid."""

    def __init__(self, strength=1.0, seed=0):
        self.strength = strength
        self.rng = np.random.default_rng(seed)

        self.rows, self.cols = np.mgrid[0:FRAME_H, 0:FRAME_W]
        self.digits = np.load(f"{OUT_DIR}/sprites.npz")["digits"]  # (10, 8, 8) masks of the HUD font
        self.reset()

    def reset(self):
        self.heat = np.zeros((ROWS, COLS), dtype=np.float32)
        self.guess = np.zeros((ROWS, COLS), dtype=np.int64)  # tiles to the left a cell is copied from
        self.jitter_y = self.offsets()
        self.jitter_x = self.offsets()

        # Each pixel's standing draw against its chance of being wrong. Keeping it from
        # step to step is what makes an error stay put for a few frames, the way it does
        # when every frame is drawn from the previous ones.
        self.luck = self.rng.random((FRAME_H, FRAME_W))
        self.stale_luck = self.rng.random((FRAME_H, FRAME_W))

        self.rot = np.zeros((ROWS, COLS), dtype=bool)  # cells that are on their way out
        self.fade = np.zeros((ROWS, COLS), dtype=np.float32)  # how far gone each cell is
        self.grain = self.grains()

        # For each HUD column: the digit shown in the top and bottom half of the cell,
        # and the digit that was really there last step. -1 where there is no digit.
        self.shown = np.full((COLS, 2), -1, dtype=np.int64)
        self.truth = np.full(COLS, -1, dtype=np.int64)

        self.last = None
        self.carry = 0  # scrolled pixels not yet a whole cell
        self.drift = 0.0

    def offsets(self):
        patches = self.rng.choice(OFFSETS, size=(FRAME_H // PATCH, FRAME_W // PATCH))

        return np.repeat(np.repeat(patches, PATCH, axis=0), PATCH, axis=1)

    def grains(self):
        chunks = self.rng.random((FRAME_H // GRAIN, FRAME_W // GRAIN))

        return np.repeat(np.repeat(chunks, GRAIN, axis=0), GRAIN, axis=1)

    def scroll(self, dx):
        # Everything the filter remembers is about a place in the level, so it moves
        # with the picture.
        self.last = np.roll(self.last, -dx, axis=1)
        self.jitter_y = np.roll(self.jitter_y, -dx, axis=1)
        self.jitter_x = np.roll(self.jitter_x, -dx, axis=1)
        self.luck = np.roll(self.luck, -dx, axis=1)
        self.stale_luck = np.roll(self.stale_luck, -dx, axis=1)
        self.grain = np.roll(self.grain, -dx, axis=1)

        self.carry += dx
        cells = self.carry // CELL
        self.carry -= cells * CELL

        if (cells > 0):
            cells = min(cells, COLS)
            self.heat = np.roll(self.heat, -cells, axis=1)
            self.guess = np.roll(self.guess, -cells, axis=1)

            self.heat[:, -cells:] = EDGE_HEAT

            self.rot = np.roll(self.rot, -cells, axis=1)
            self.fade = np.roll(self.fade, -cells, axis=1)
            self.rot[:, -cells:] = False
            self.fade[:, -cells:] = 0

            # The guess is made for 16-pixel blocks, so what gets copied is whole tiles.
            blocks = (self.rng.random((ROWS // 2, cells)) < GUESS_CHANCE) * self.rng.integers(1, 5, size=(ROWS // 2, cells))
            self.guess[:, -cells:] = np.repeat(blocks, 2, axis=0)

    def warm(self, boxes):
        rng = self.rng

        self.drift = float(np.clip(self.drift + rng.normal(0, DRIFT_STEP), 0, DRIFT_MAX))
        self.heat = self.heat * COOL + BASE_HEAT + self.drift

        for x, y, width, height in boxes:
            r0, r1 = max(y // CELL, 0), min((y + height - 1) // CELL + 1, ROWS)
            c0, c1 = max(x // CELL, 0), min((x + width - 1) // CELL + 1, COLS)
            self.heat[r0:r1, c0:c1] += MOTION_HEAT

        if (rng.random() < FLARE_CHANCE):
            r, c = rng.integers(ROWS), rng.integers(COLS)
            self.heat[max(r - 2, 0):r + 3, max(c - 3, 0):c + 4] += FLARE_HEAT

        self.heat[HUD_ROWS] = np.maximum(self.heat[HUD_ROWS], HUD_HEAT)

        padded = np.pad(self.heat, 1, mode="edge")
        around = sum(padded[1 + dr:1 + dr + ROWS, 1 + dc:1 + dc + COLS] for dr in (-1, 0, 1) for dc in (-1, 0, 1)) / 9
        self.heat = np.clip((1 - SPREAD) * self.heat + SPREAD * around, 0, 1)

    def erode(self, frame, out, boxes):
        # Small details phase out of existence and the inside of big things melts.
        rng = self.rng
        backdrop = frame[2, 2]

        # cells shape: (28, 32, 64) -- the pixels of each latent cell
        cells = frame.reshape(ROWS, CELL, COLS, CELL).transpose(0, 2, 1, 3).reshape(ROWS, COLS, CELL * CELL)
        empty = (cells == backdrop).mean(axis=2)

        padded = np.pad(empty, 1, mode="edge")
        around = sum(padded[1 + dr:1 + dr + ROWS, 1 + dc:1 + dc + COLS] for dr in (-1, 0, 1) for dc in (-1, 0, 1)) / 9
        alone = (around > ALONE)

        # Rot starts in a cell with something in it, creeps to its neighbours, and after
        # a while stops; then the cell slowly comes back.
        spreading = np.pad(self.rot, 1)
        touched = spreading[:-2, 1:-1] | spreading[2:, 1:-1] | spreading[1:-1, :-2] | spreading[1:-1, 2:]

        self.rot |= (rng.random((ROWS, COLS)) < ROT_CHANCE * self.strength) & (empty < 0.98)
        self.rot |= touched & (rng.random((ROWS, COLS)) < ROT_SPREAD * self.strength) & (empty < 0.98)
        self.rot &= (rng.random((ROWS, COLS)) >= ROT_STOP)
        self.rot[:HUD_ROWS.stop] = False

        # Whatever Mario is next to is put back, the way the real model redraws what the
        # player interacts with. Mario is the last box the engine lists.
        if (len(boxes) > 0):
            x, y, width, height = boxes[-1]
            r0, r1 = max(y // CELL - NEAR_MARIO[0], 0), (y + height) // CELL + NEAR_MARIO[0] + 1
            c0, c1 = max(x // CELL - NEAR_MARIO[1], 0), (x + width) // CELL + NEAR_MARIO[1] + 1
            self.rot[r0:r1, c0:c1] = False
            self.fade[r0:r1, c0:c1] *= 0.6

        self.fade = np.clip(self.fade + np.where(self.rot, FADE_RATE, -FADE_BACK), 0, 1)

        # A small thing on its own dissolves into the backdrop. A cell inside something
        # big keeps its main colour and loses its lines. The median of a cell's colour
        # numbers is a colour that is in the cell, and nearly always its commonest.
        main = np.sort(cells, axis=2)[:, :, CELL * CELL // 2]
        fill = np.where(alone, backdrop, main).astype(np.uint8)
        gone = np.where(alone, self.fade, self.fade * MELT_MAX)

        gone = np.repeat(np.repeat(gone, CELL, axis=0), CELL, axis=1)
        fill = np.repeat(np.repeat(fill, CELL, axis=0), CELL, axis=1)

        return np.where(self.grain < gone, fill, out)

    def misread(self, frame, out):
        # The HUD digits: sharp, but not always the right ones. The clock is the one
        # that shows it most, because it has to change: it stalls, then jumps.
        rng = self.rng
        backdrop = frame[2, 2]

        # lit shape: (32, 8, 8) -- the text pixels of each cell on the digit row
        row = frame[DIGIT_Y:DIGIT_Y + CELL]
        lit = (row.reshape(CELL, COLS, CELL).transpose(1, 0, 2) != backdrop)
        matches = (lit[:, None] == self.digits[None]).all(axis=(2, 3))

        for col in np.flatnonzero(matches.any(axis=1)):
            truth = int(matches[col].argmax())
            top, bottom = self.shown[col]
            right = (top == truth and bottom == truth)

            if (top < 0):
                top = bottom = truth
            elif (truth != self.truth[col]):
                if (rng.random() >= DIGIT_STALL * min(self.strength, 2)):
                    top = bottom = truth
            elif (not right):
                if (rng.random() < DIGIT_HEAL):
                    top = bottom = truth
            elif (rng.random() < DIGIT_MUTATE * self.strength):
                other = int(rng.choice(LOOKALIKES[truth]))
                top, bottom = (other, other) if (rng.random() >= DIGIT_HYBRID) else (truth, other)

            self.shown[col] = (top, bottom)
            self.truth[col] = truth

            if (top != truth or bottom != truth):
                glyph = np.concatenate([self.digits[top][:CELL // 2], self.digits[bottom][CELL // 2:]])
                color = row[:, col * CELL:(col + 1) * CELL][lit[col]][0]
                out[DIGIT_Y:DIGIT_Y + CELL, col * CELL:(col + 1) * CELL] = np.where(glyph, color, backdrop)

        gone = ~matches.any(axis=1)
        self.shown[gone] = -1
        self.truth[gone] = -1

        return out

    def step(self, frame, dx=0, boxes=(), cut=False):
        # frame shape: (224, 256) NES colour numbers, exact -> the same, decayed.
        # dx is how far the picture scrolled since the last frame; boxes are the
        # (x, y, width, height) of what moves; cut says the scene was replaced, which
        # is a fresh start for a world model too (it is re-seeded with real frames).
        if (cut or self.last is None):
            self.reset()
            self.last = frame.copy()

            return frame

        rng = self.rng

        self.scroll(dx)
        self.warm(boxes)

        # The roll wrapped the left edge round to the right; the last frame has
        # nothing to say about columns that were not in it.
        if (dx > 0):
            self.last[:, -dx:] = frame[:, -dx:]

        # p shape: (224, 256) -- each pixel's chance of being wrong this step
        p = np.repeat(np.repeat(self.heat, CELL, axis=0), CELL, axis=1) * self.strength

        # Newly seen scenery is first drawn as a copy of scenery already on screen.
        shift = np.repeat(np.repeat(self.guess, CELL, axis=0), CELL, axis=1) * 2 * CELL
        out = frame[self.rows, np.clip(self.cols - shift * (self.strength > 0), 0, FRAME_W - 1)]
        self.guess[rng.random((ROWS, COLS)) < GUESS_HEAL] = 0

        # A wrong pixel takes the colour of a pixel a step or two away. Inside a flat
        # area that is the same colour, so only edges and texture visibly crawl.
        redraw = (rng.random((FRAME_H, FRAME_W)) < RESAMPLE)
        self.jitter_y = np.where(redraw, self.offsets(), self.jitter_y)
        self.jitter_x = np.where(redraw, self.offsets(), self.jitter_x)
        self.luck = np.where(redraw, rng.random((FRAME_H, FRAME_W)), self.luck)
        self.stale_luck = np.where(redraw, rng.random((FRAME_H, FRAME_W)), self.stale_luck)

        near = out[np.clip(self.rows + self.jitter_y, 0, FRAME_H - 1), np.clip(self.cols + self.jitter_x, 0, FRAME_W - 1)]
        wrong = (self.luck < p * JITTER)
        stale = (self.stale_luck < p * LAG)

        out = np.where(wrong, near, out)
        out = np.where(stale, self.last, out)

        if (self.strength > 0):
            out = self.erode(frame, out, boxes)
            out = self.misread(frame, out)

        self.last = out

        return out
