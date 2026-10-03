from envs.mario import BUTTONS, FRAME_H, FRAME_W, FRAME_SKIP
from simul.build_assets import OUT_DIR, EMPTY, TILE, TILE_ROWS, TILE_TOP, MARIO_WINDOW, ENEMY_WINDOW, DIGIT_Y, TIMER_X, WATER
import numpy as np

NUM_LEVELS = 32
START_LIVES = 3

# Mario's movement, in pixels per NES frame. These are the original game's numbers, so
# a jump clears the same gaps it does there; the levels were built around them.
WALK_MAX = 1.5625
RUN_MAX = 2.5625
WALK_ACCEL = 0.0371
RUN_ACCEL = 0.0557
RELEASE = 0.0508  # slowing down with no direction held
SKID = 0.1016  # slowing down with the opposite direction held
JUMP_SPEED = 4.0
RUNNING_JUMP_SPEED = 5.0  # a jump from a full run starts faster
RUNNING_JUMP_AT = 2.3125
HOLD_GRAVITY = (0.125, 0.1172, 0.1563)  # while A is held on the way up; by speed at take-off
FALL_GRAVITY = (0.4375, 0.375, 0.5625)
MAX_FALL = 4.0
STOMP_BOUNCE = 4.0

SWIM_STROKE = 1.5
SWIM_GRAVITY = 0.04
SWIM_MAX_FALL = 1.5
SWIM_MAX = 1.0

ENEMY_SPEED = 0.5
ENEMY_GRAVITY = 0.3
MUSHROOM_SPEED = 1.0

SCROLL_AT = 112  # Mario's screen x past which the camera follows him; it never goes back
BOX_LEFT = 3  # Mario's hitbox inside his 16x32 box
BOX_RIGHT = 13
SMALL_TOP = 19
BIG_TOP = 7

TIMER_START = 400
TIMER_FRAMES = 24  # NES frames per tick of the clock
HURT_FRAMES = 120  # flashing and untouchable after a hit
DYING_FRAMES = 150
CLEAR_FRAMES = 40
SQUASH_FRAMES = 30

# Tile numbers, the game's own. Everything else that is not 0 is solid.
BRICKS = (0x51, 0x52)
POWERUP_BLOCKS = (0xC1, 0x55, 0x5A)
ITEM_BLOCKS = (0xC0, 0xC1) + tuple(range(0x55, 0x5F))
HIDDEN = (0x5F, 0x60)  # solid only from below
COINS = (0xC2, 0xC3)
FLAGPOLE = (0x24, 0x25)
VINE = 0x26
USED = 0xC4
PASSABLE = (0,) + HIDDEN + COINS + FLAGPOLE + (VINE,)

# Which drawing of Mario goes with which pose: the game's graphics offsets, read off
# the table build_assets prints. Small Mario has his own set; big and fire share one.
POSES = {
    False: {"walk": (0x60, 0x68, 0x70), "skid": 0x78, "jump": 0x80, "swim": (0x88, 0x90, 0x98), "climb": 0xA0, "die": 0xB0, "stand": 0xB8, "crouch": 0xB8},
    True: {"walk": (0x00, 0x08, 0x10), "skid": 0x18, "jump": 0x20, "swim": (0x28, 0x30, 0x38), "climb": 0x40, "die": 0x20, "stand": 0xC8, "crouch": 0x50}
}

GOOMBA = 0x06
SAME_LOOK = {0x01: 0x03, 0x04: 0x00}  # koopa kinds that only differ in how they walk
MUSHROOM = 0x100
FLOWER = 0x101

A, B = BUTTONS.index("A"), BUTTONS.index("B")
LEFT, RIGHT, DOWN = BUTTONS.index("LEFT"), BUTTONS.index("RIGHT"), BUTTONS.index("DOWN")


class Sprites:
    def __init__(self):
        data = np.load(f"{OUT_DIR}/sprites.npz")

        self.mario = {int(key): image for key, image in zip(data["mario_keys"], data["mario"])}
        self.objects = {tuple(int(v) for v in key): image for key, image in zip(data["object_keys"], data["objects"])}
        self.blocks = {tuple(int(v) for v in key): image for key, image in zip(data["block_keys"], data["blocks"])}
        self.digits = data["digits"]
        self.palette = data["palette"]

    def object(self, kind, area_type, phase):
        # Not every enemy was seen in every kind of area; the same enemy from another
        # area has the wrong colours, which still beats a missing one.
        kind = SAME_LOOK.get(kind, kind)

        for area in [area_type, 1, 2, 3, 0]:
            if ((kind, area, phase) in self.objects):
                return self.objects[(kind, area, phase)]

        return None

    def block(self, level, tile):
        if ((level, tile) in self.blocks):
            return self.blocks[(level, tile)]

        for (other, kind), image in self.blocks.items():
            if (kind == tile):
                return image

        return None


def extent(sprite):
    # (top, bottom, left, right) of the drawn pixels inside a sprite's window
    rows = np.flatnonzero((sprite != EMPTY).any(axis=1))
    cols = np.flatnonzero((sprite != EMPTY).any(axis=0))

    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


class Game:
    """Super Mario Bros rebuilt from the recorded frames: backgrounds stitched from them,
    sprites cut out of them, solid tiles and enemy positions read from the recorded RAM.
    step() takes the buttons and returns the next frame, the same contract as
    Dreamer.step, but nothing is generated: this is ordinary game code."""

    def __init__(self, level=0):
        self.sprites = Sprites()
        self.lives = START_LIVES
        self.score = 0
        self.coins = 0
        self.big = False
        self.fire = False

        self.load(level)

    def load(self, level, piece=0):
        data = np.load(f"{OUT_DIR}/level_{level:02d}.npz")

        self.level = level
        self.background = data["background"].copy()
        self.tiles = data["tiles"].copy()
        self.hud = data["hud"]
        self.pieces = data["pieces"]
        self.backdrop = int(data["backdrop"])
        self.area_type = int(data["area_type"])
        self.water = (self.area_type == WATER)

        # Different runs first saw the same enemy a pixel or two apart, so sightings of
        # one kind at one height within a few pixels of each other are one enemy.
        self.spawns = []

        for kind, x, y in data["spawns"].tolist():
            if (not any(kind == other and y == height and abs(x - place) <= 6 for other, place, height in self.spawns)):
                self.spawns.append((kind, x, y))

        self.spawned = [False] * len(self.spawns)

        # paths[i] shape: (frames, 2) -- where platform i was on each recorded frame
        ends = np.cumsum(data["lift_lengths"])
        self.lift_kinds = [int(kind) for kind in data["lift_kinds"]]
        self.lift_paths = [data["lift_paths"][end - length:end].astype(np.int64) for end, length in zip(ends, data["lift_lengths"])]
        self.lift_started = [False] * len(self.lift_kinds)

        # The HUD text is the lightest thing on the top rows.
        text = self.hud[self.hud != self.backdrop]
        self.text_color = int(np.bincount(text).argmax()) if (len(text) > 0) else 0x30

        # The stitched background shows each block the way most recordings left it:
        # question blocks used, bricks broken, coins taken. The tile grid was read
        # before anyone touched them, so they are drawn back in from it.
        for row, col in zip(*np.nonzero(np.isin(self.tiles, BRICKS + ITEM_BLOCKS + COINS))):
            self.set_tile(col, row, self.tiles[row, col])

        self.timer = TIMER_START
        self.frame = 0
        self.enter(piece)

    def enter(self, piece):
        self.piece = piece
        left, right, x, y, reach = [int(v) for v in self.pieces[piece]]

        self.x, self.y = float(x), float(y)
        self.vx, self.vy = 0.0, 0.0
        self.facing = 1
        self.grounded = False
        self.gravity_class = 0
        self.crouching = False
        self.walk_distance = 0.0
        self.riding = None

        self.scroll = int(np.clip(x - SCROLL_AT, left, max(right - FRAME_W, left)))
        self.enemies = []
        self.items = []
        self.lifts = []

        self.mode = "play"
        self.mode_frames = 0
        self.hurt_frames = 0
        self.jump_held = True  # a button held through the cut must be pressed again
        self.cut = True  # tells the viewer that this frame does not follow from the last

    def solid(self, col, row, from_below=False):
        if (col < 0):
            return True
        if (row < 0 or row >= TILE_ROWS or col >= self.tiles.shape[1]):
            return False

        tile = self.tiles[row, col]

        if (from_below and tile in HIDDEN):
            return True

        return (tile not in PASSABLE)

    def set_tile(self, col, row, tile):
        self.tiles[row, col] = tile
        image = self.sprites.block(self.level, tile) if (tile != 0) else None
        y, x = TILE_TOP + row * TILE, col * TILE

        self.background[y:y + TILE, x:x + TILE] = self.backdrop if (image is None) else image[:FRAME_H - y]

    def bump(self, col, row):
        tile = self.tiles[row, col]

        if (tile in BRICKS):
            if (self.big):
                self.set_tile(col, row, 0)
                self.score += 50

        elif (tile in ITEM_BLOCKS or tile in HIDDEN):
            self.set_tile(col, row, USED)

            if (tile in POWERUP_BLOCKS):
                kind = FLOWER if (self.big) else MUSHROOM
                self.items.append({"kind": kind, "x": float(col * TILE), "y": float(TILE_TOP + (row - 1) * TILE - 8), "vx": MUSHROOM_SPEED if (kind == MUSHROOM) else 0.0, "vy": 0.0})
            else:
                self.coins += 1
                self.score += 200

        # Anything standing on the block when it is hit from below is knocked off.
        for enemy in self.enemies:
            if (abs(enemy["x"] - col * TILE) < 12 and abs(enemy["y"] + 24 - (TILE_TOP + row * TILE)) < 4):
                enemy["squashed"] = SQUASH_FRAMES
                self.score += 100

    def top(self):
        return BIG_TOP if (self.big and not self.crouching) else SMALL_TOP

    def move_x(self, dx):
        self.x += dx
        left, right = self.x + BOX_LEFT, self.x + BOX_RIGHT - 0.01

        rows = range(int((self.y + self.top() - TILE_TOP) // TILE), int((self.y + 31 - TILE_TOP) // TILE) + 1)

        if (dx > 0 and any(self.solid(int(right // TILE), row) for row in rows)):
            self.x = int(right // TILE) * TILE - BOX_RIGHT
            self.vx = 0.0

        if (dx < 0 and any(self.solid(int(left // TILE), row) for row in rows)):
            self.x = (int(left // TILE) + 1) * TILE - BOX_LEFT
            self.vx = 0.0

        # The camera never scrolls back, and the left edge of the screen is a wall.
        if (self.x < self.scroll):
            self.x = float(self.scroll)
            self.vx = max(self.vx, 0.0)

    def move_y(self, dy):
        before = self.y + 32
        self.y += dy
        self.grounded = False

        cols = (int((self.x + BOX_LEFT + 1) // TILE), int((self.x + BOX_RIGHT - 1) // TILE))

        if (dy >= 0):
            row = int((self.y + 32 - TILE_TOP) // TILE)
            tile_top = TILE_TOP + row * TILE

            # Only a floor Mario came down onto from above counts, so that being pushed
            # into the side of a wall does not pop him up on top of it.
            if (any(self.solid(col, row) for col in cols) and before <= tile_top + MAX_FALL + 1):
                self.y = float(tile_top - 32)
                self.vy = 0.0
                self.grounded = True
        else:
            row = int((self.y + self.top() - TILE_TOP) // TILE)
            col = int((self.x + 8) // TILE)

            # The head is tested at its middle, like the game does, so Mario slides past
            # the corner of a block instead of catching on it.
            if (self.solid(col, row, from_below=True)):
                self.y = float(TILE_TOP + (row + 1) * TILE - self.top())
                self.vy = 1.0
                self.bump(col, row)

    def control(self, action, jump_pressed):
        direction = int(action[RIGHT]) - int(action[LEFT])
        self.crouching = bool(action[DOWN]) and self.big and self.grounded

        if (self.crouching):
            direction = 0

        if (self.water):
            top_speed, accel = SWIM_MAX, WALK_ACCEL
        else:
            top_speed, accel = (RUN_MAX, RUN_ACCEL) if (action[B]) else (WALK_MAX, WALK_ACCEL)

        if (direction != 0):
            if (self.grounded):
                self.facing = direction

            # Pushing against the way Mario is moving is a skid, which brakes harder.
            if (self.vx * direction < 0):
                self.vx += direction * SKID
            elif (abs(self.vx) < top_speed):
                self.vx += direction * accel
            elif (self.grounded):
                self.vx -= np.sign(self.vx) * RELEASE
        elif (self.grounded or self.water):
            self.vx -= np.sign(self.vx) * min(RELEASE, abs(self.vx))

        if (self.water):
            if (jump_pressed):
                self.vy = -SWIM_STROKE

            self.vy = min(self.vy + SWIM_GRAVITY, SWIM_MAX_FALL)

            if (self.y < TILE_TOP - 8):
                self.y = float(TILE_TOP - 8)
                self.vy = max(self.vy, 0.0)

            return

        if (jump_pressed and self.grounded):
            speed = abs(self.vx)
            self.vy = -(RUNNING_JUMP_SPEED if (speed >= RUNNING_JUMP_AT) else JUMP_SPEED)
            self.gravity_class = 0 if (speed < 1.0) else (1 if (speed < RUNNING_JUMP_AT) else 2)
            self.grounded = False
            self.riding = None

        # Holding A on the way up is what makes a jump high: gravity is weaker for as
        # long as it is held.
        rising = (self.vy < 0 and action[A])
        gravity = HOLD_GRAVITY[self.gravity_class] if (rising) else FALL_GRAVITY[self.gravity_class]
        self.vy = min(self.vy + gravity, MAX_FALL)

    def touch_tiles(self):
        # Coins and the flagpole are tiles Mario overlaps instead of bumping into.
        for px, py in [(self.x + 8, self.y + self.top() + 2), (self.x + 8, self.y + 28)]:
            col, row = int(px // TILE), int((py - TILE_TOP) // TILE)

            if (row < 0 or row >= TILE_ROWS or col >= self.tiles.shape[1]):
                continue

            tile = self.tiles[row, col]

            if (tile in COINS):
                self.set_tile(col, row, 0)
                self.coins += 1
                self.score += 200

            if (tile in FLAGPOLE and self.mode == "play"):
                self.mode = "flag"
                self.x = float(col * TILE - 6)
                self.vx = self.vy = 0.0
                self.facing = 1
                self.score += 1000

    def spawn(self):
        for i, (kind, x, y) in enumerate(self.spawns):
            if (not self.spawned[i] and self.scroll <= x <= self.scroll + FRAME_W + TILE):
                self.spawned[i] = True
                self.enemies.append({"kind": kind, "x": float(x), "y": float(y), "vx": -ENEMY_SPEED, "vy": 0.0, "squashed": 0})

        for i, path in enumerate(self.lift_paths):
            if (not self.lift_started[i] and self.scroll - 64 <= path[0, 0] <= self.scroll + FRAME_W + TILE):
                self.lift_started[i] = True
                sprite = self.sprites.object(self.lift_kinds[i], self.area_type, 0)

                if (sprite is not None):
                    self.lifts.append({"kind": self.lift_kinds[i], "path": path, "t": 0, "step": 1, "x": int(path[0, 0]), "y": int(path[0, 1]), "extent": extent(sprite)})

    def walker(self, thing, turn):
        # Shared by enemies and the mushroom: walk, fall, turn round at a wall.
        thing["vy"] = min(thing["vy"] + ENEMY_GRAVITY, MAX_FALL)
        thing["y"] += thing["vy"]

        row = int((thing["y"] + 24 - TILE_TOP) // TILE)

        if (self.solid(int((thing["x"] + 4) // TILE), row) or self.solid(int((thing["x"] + 12) // TILE), row)):
            thing["y"] = float(TILE_TOP + row * TILE - 24)
            thing["vy"] = 0.0

        thing["x"] += thing["vx"]
        ahead = thing["x"] + (14 if (thing["vx"] > 0) else 2)

        if (turn and self.solid(int(ahead // TILE), int((thing["y"] + 16 - TILE_TOP) // TILE))):
            thing["vx"] = -thing["vx"]
            thing["x"] += 2 * thing["vx"]

    def hurt(self):
        if (self.hurt_frames > 0):
            return

        if (self.big):
            self.big = self.fire = False
            self.hurt_frames = HURT_FRAMES
        else:
            self.die(hop=True)

    def die(self, hop):
        self.mode = "dying"
        self.mode_frames = 0
        self.big = self.fire = False
        self.vx = 0.0
        self.vy = -JUMP_SPEED if (hop) else 0.0

    def objects(self):
        for lift in self.lifts:
            # The platform replays its recorded path, back and forth, and carries
            # whatever stands on it.
            path = lift["path"]

            if (len(path) > 1):
                if (not 0 <= lift["t"] + lift["step"] < len(path)):
                    lift["step"] = -lift["step"]
                lift["t"] += lift["step"]

            x, y = int(path[lift["t"], 0]), int(path[lift["t"], 1])
            dx, dy = x - lift["x"], y - lift["y"]
            lift["x"], lift["y"] = x, y

            top, bottom, left, right = lift["extent"]
            surface = y - 16 + top
            feet = self.y + 32

            if (self.riding is lift):
                # A recorded platform can jump (it wraps round the screen); Mario is
                # only carried by movement he could have kept up with.
                if (abs(dx) > 8 or abs(dy) > 8 or not (x - 8 + left - BOX_RIGHT < self.x < x - 8 + right - BOX_LEFT)):
                    self.riding = None
                else:
                    self.move_x(dx)
                    self.y = float(surface - 32)
                    self.vy = 0.0
                    self.grounded = True

            elif (self.mode == "play" and self.vy >= 0 and surface - 2 <= feet <= surface + 6 + abs(dy) and x - 8 + left - BOX_RIGHT < self.x < x - 8 + right - BOX_LEFT):
                self.riding = lift
                self.y = float(surface - 32)
                self.vy = 0.0
                self.grounded = True

        for enemy in self.enemies:
            if (enemy["squashed"] > 0):
                enemy["squashed"] -= 1
                continue

            self.walker(enemy, turn=True)

            if (self.mode != "play"):
                continue

            overlap = (abs(enemy["x"] - self.x) < 12 and self.y + self.top() < enemy["y"] + 24 and self.y + 32 > enemy["y"] + 10)

            if (overlap):
                # Coming down onto an enemy is a stomp; touching it any other way hurts.
                if (self.vy > 0 and self.y + 32 < enemy["y"] + 22):
                    enemy["squashed"] = SQUASH_FRAMES
                    self.vy = -STOMP_BOUNCE
                    self.score += 100
                else:
                    self.hurt()

        self.enemies = [
            enemy for enemy in self.enemies
            if (enemy["y"] < FRAME_H + 32 and enemy["x"] > self.scroll - 48 and enemy["squashed"] != 1)
        ]

        for item in self.items:
            if (item["kind"] == MUSHROOM):
                self.walker(item, turn=True)

            if (self.mode == "play" and abs(item["x"] - self.x) < 12 and abs(item["y"] + 8 - self.y - 16) < 20):
                item["kind"] = None
                self.score += 1000

                if (self.big):
                    self.fire = True
                self.big = True

        self.items = [item for item in self.items if (item["kind"] is not None and item["y"] < FRAME_H + 32)]

    def tick(self, action, jump_pressed):
        # One NES frame.
        self.frame += 1
        self.mode_frames += 1
        self.hurt_frames = max(self.hurt_frames - 1, 0)

        left, right, entry_x, entry_y, reach = [int(v) for v in self.pieces[self.piece]]
        last_piece = (self.piece == len(self.pieces) - 1)

        if (self.mode == "play"):
            if (self.frame % TIMER_FRAMES == 0):
                self.timer = max(self.timer - 1, 0)

            self.control(action, jump_pressed)

            if (self.riding is None):
                self.move_x(self.vx)
                self.move_y(self.vy)
            else:
                self.move_x(self.vx)

            self.walk_distance += abs(self.vx)
            self.touch_tiles()

            if (self.y > FRAME_H + 16):
                self.die(hop=False)

            elif (self.mode == "play" and self.x >= reach - 2):
                # The end of what the recordings show. In the middle of a level that
                # is a pipe every player took, so the game cuts to where it came out.
                if (last_piece):
                    self.mode = "clear"
                    self.mode_frames = 0
                else:
                    self.enter(self.piece + 1)
                    return

        elif (self.mode == "flag"):
            # Slide down the pole, then walk off to the castle.
            self.y += 2.0
            row = int((self.y + 32 - TILE_TOP) // TILE)

            if (self.solid(int((self.x + 8) // TILE), row) or self.solid(int((self.x + 24) // TILE), row) or self.y > FRAME_H - 56):
                self.y = min(float(TILE_TOP + row * TILE - 32), self.y)
                self.mode = "walkoff"
                self.x += 12

        elif (self.mode == "walkoff"):
            self.vx = WALK_MAX
            self.vy = min(self.vy + FALL_GRAVITY[0], MAX_FALL)
            self.move_x(self.vx)
            self.move_y(self.vy)
            self.walk_distance += self.vx

            if (self.x >= reach - 2 or self.mode_frames > 400):
                self.mode = "clear"
                self.mode_frames = 0

        elif (self.mode == "clear"):
            if (self.mode_frames >= CLEAR_FRAMES):
                self.load((self.level + 1) % NUM_LEVELS)
                return

        elif (self.mode == "dying"):
            self.vy = min(self.vy + HOLD_GRAVITY[0], MAX_FALL)
            self.y += self.vy

            if (self.mode_frames >= DYING_FRAMES):
                self.lives -= 1

                if (self.lives == 0):
                    self.lives = START_LIVES
                    self.score = 0
                    self.coins = 0
                    self.load(0)
                else:
                    self.load(self.level, self.piece)

                return

        # The camera follows Mario to the right and stops at the end of what was seen.
        self.scroll = int(np.clip(max(self.scroll, self.x - SCROLL_AT), left, max(right - FRAME_W, left)))

        self.spawn()
        self.objects()

    def step(self, action):
        # action shape: (6,) in BUTTONS order -> frame shape: (224, 256) NES colour numbers.
        # The buttons are held for FRAME_SKIP NES frames, as in envs.mario.step.
        jump_pressed = bool(action[A]) and not self.jump_held
        self.jump_held = bool(action[A])
        self.cut = False

        for i in range(FRAME_SKIP):
            self.tick(action, jump_pressed and i == 0)

        return self.render()

    def pose(self):
        poses = POSES[self.big]

        if (self.mode == "dying"):
            return poses["die"]
        if (self.mode == "flag"):
            return poses["climb"]
        if (self.water and not self.grounded):
            return poses["swim"][(self.frame // 8) % 3]
        if (not self.grounded):
            return poses["jump"]
        if (self.crouching):
            return poses["crouch"]
        if (abs(self.vx) < 0.05):
            return poses["stand"]
        if (self.vx * self.facing < 0):
            return poses["skid"]

        return poses["walk"][int(self.walk_distance / 5) % 3]

    def draw(self, frame, sprite, x, y):
        # sprite shape: (H, W) with EMPTY for see-through; x, y is its top left on screen
        H, W = sprite.shape
        y0, y1 = max(y, 0), min(y + H, FRAME_H)
        x0, x1 = max(x, 0), min(x + W, FRAME_W)

        if (y1 <= y0 or x1 <= x0):
            return

        piece = sprite[y0 - y:y1 - y, x0 - x:x1 - x]
        target = frame[y0:y1, x0:x1]
        target[piece != EMPTY] = piece[piece != EMPTY]

    def number(self, frame, value, digits, x):
        for i, digit in enumerate(f"{value:0{digits}d}"[-digits:]):
            cell = frame[DIGIT_Y:DIGIT_Y + 8, x + 8 * i:x + 8 * i + 8]
            cell[:] = np.where(self.sprites.digits[int(digit)], self.text_color, self.backdrop)

    def render(self):
        frame = self.background[:, self.scroll:self.scroll + FRAME_W].copy()

        if (frame.shape[1] < FRAME_W):
            frame = np.pad(frame, ((0, 0), (0, FRAME_W - frame.shape[1])), constant_values=self.backdrop)

        self.boxes = []  # (x, y, width, height) on screen of everything that moves
        phase = (self.frame >> 3) & 1

        for lift in self.lifts:
            sprite = self.sprites.object(lift["kind"], self.area_type, 0)
            self.draw(frame, sprite, lift["x"] - 8 - self.scroll, lift["y"] - 16)
            self.boxes.append((lift["x"] - self.scroll, lift["y"], 48, 8))

        for item in self.items:
            sprite = self.sprites.object(item["kind"], self.area_type, phase)

            if (sprite is None):
                sprite = self.sprites.object(MUSHROOM, self.area_type, 0)
            if (sprite is not None):
                self.draw(frame, sprite, int(item["x"]) - 8 - self.scroll, int(item["y"]) - 16)

            self.boxes.append((int(item["x"]) - self.scroll, int(item["y"]) + 8, 16, 16))

        for enemy in self.enemies:
            sprite = self.sprites.object(enemy["kind"], self.area_type, phase)

            if (sprite is None):
                sprite = self.sprites.object(GOOMBA, self.area_type, phase)
            if (sprite is None):
                continue

            if (enemy["squashed"] > 0):
                # A stomped enemy is its own drawing pressed flat onto the ground.
                top, bottom, left, right = extent(sprite)
                flat = np.full_like(sprite, EMPTY)
                rows = np.linspace(top, bottom - 1, (bottom - top + 1) // 2).astype(np.int64)
                flat[bottom - len(rows):bottom] = sprite[rows]
                sprite = flat

            if (enemy["vx"] > 0):
                sprite = sprite[:, ::-1]
                self.draw(frame, sprite, int(enemy["x"]) + 24 - ENEMY_WINDOW[1] - self.scroll, int(enemy["y"]) - 16)
            else:
                self.draw(frame, sprite, int(enemy["x"]) - 8 - self.scroll, int(enemy["y"]) - 16)

            self.boxes.append((int(enemy["x"]) - self.scroll, int(enemy["y"]) + 8, 16, 16))

        # Mario flashes while he cannot be hurt, and is gone once the level is over.
        visible = (self.mode != "clear" and (self.hurt_frames // 4) % 2 == 0)
        size = 2 if (self.fire) else (1 if (self.big) else 0)
        sprite = self.sprites.mario.get(size * 256 + self.pose())

        if (visible and sprite is not None):
            if (self.facing < 0):
                sprite = sprite[:, ::-1]

            self.draw(frame, sprite, int(self.x) - 8 - self.scroll, int(self.y) - 8)

        self.boxes.append((int(self.x) - self.scroll, int(self.y), 16, 32))

        # The HUD sits on top of everything and does not scroll.
        text = (self.hud != self.backdrop)
        frame[:TILE_TOP][text] = self.hud[text]
        self.number(frame, self.score, 6, 24)
        self.number(frame, self.coins % 100, 2, 104)
        self.number(frame, self.timer, 3, TIMER_X)

        return frame
