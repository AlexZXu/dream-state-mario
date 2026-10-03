from envs.mario import FRAME_H, FRAME_W, CROP_TOP, RAM_PLAYER_STATE, RAM_X_PAGE, RAM_X_ON_PAGE, RAM_Y_ON_SCREEN, RAM_POWERUP
from envs.smb import load, NUM_COLORS
from envs.smb_events import RAM_Y_SCREEN, FLAGPOLE
import os

import numpy as np

OUT_DIR = "data/replica"

# More addresses in the console's RAM, on top of the ones in envs/mario.py. All of them
# were checked against the frames by looking at what build_assets cuts out with them.
RAM_FRAME_COUNTER = 0x0009
RAM_ENEMY_ACTIVE = 0x000F  # 5 enemy slots, then one slot for the power-up
RAM_ENEMY_TYPE = 0x0016
RAM_FLOAT_STATE = 0x001D  # 0 on the ground, 1 jumping, 2 falling
RAM_ENEMY_STATE = 0x001E
RAM_FACING = 0x0033  # 1 right, 2 left
RAM_POWERUP_TYPE = 0x0039  # 0 mushroom, 1 flower, 2 star, 3 one-up
RAM_MOVING_DIR = 0x0045
RAM_ENEMY_DIR = 0x0046
RAM_X_SPEED = 0x0057
RAM_ENEMY_X_PAGE = 0x006E
RAM_ENEMY_X = 0x0087
RAM_ENEMY_Y_SCREEN = 0x00B6
RAM_ENEMY_Y = 0x00CF
RAM_PLAYER_REL_X = 0x03AD  # Mario's x on the screen
RAM_TILES = 0x0500  # two 16x13 pages of solid tiles: even level pages, then odd ones
RAM_GFX_OFFSET = 0x06D5  # which drawing of Mario is on screen
RAM_CROUCHING = 0x0714
RAM_SCROLL_PAGE = 0x071A
RAM_SCROLL_X = 0x071C
RAM_AREA_TYPE = 0x074E  # 0 water, 1 overworld, 2 underground, 3 castle
RAM_AREA_POINTER = 0x0750
RAM_SMALL = 0x0754
RAM_INJURY_TIMER = 0x079E
RAM_STAR_TIMER = 0x079F
RAM_TIMER_DIGITS = 0x07F8

ENTERING = 0x07  # player state while an area is loading
INTRO_POINTER = 0x29  # the cutscene before 1-2, 2-2, 4-2 and 7-2 where Mario walks into a pipe
WATER = 0

TILE = 16
TILE_ROWS = 13
TILE_TOP = 32 - CROP_TOP  # screen row of the first tile row
PAGE_BYTES = TILE_ROWS * 16

EMPTY = NUM_COLORS  # the "see-through" colour number in a sprite
VOTE_FRAMES = 12000  # frames per level that vote on its background; more only costs disk reads
GAP = 32  # a stretch of level narrower than this with no frames is not treated as a hole

WALKERS = [0x00, 0x01, 0x02, 0x03, 0x04, 0x06]  # koopas, buzzy beetle, goomba
LIFTS = list(range(0x24, 0x2D))  # moving platforms
POWERUP_OBJECT = 0x2E
SLOTS = 6

MARIO_WINDOW = (48, 32)  # height, width of the cut-out around Mario's 16x32 box
ENEMY_WINDOW = (40, 64)  # wide enough for the longest platform
SAMPLES = 150  # cut-outs voted on per sprite

DIGIT_Y = 16
TIMER_X = 208


def level_frames(ram, episode):
    # Returns the frames of the episode that show the level's main area. A level is
    # several areas (bonus rooms, the pipe cutscene), each with its own scroll, and only
    # the main one is rebuilt.
    start = int(episode["start"])
    end = start + int(episode["length"])

    state = ram[start:end, RAM_PLAYER_STATE]
    scroll = ram[start:end, RAM_SCROLL_PAGE].astype(np.int64) * 256 + ram[start:end, RAM_SCROLL_X]
    area_type = ram[start:end, RAM_AREA_TYPE]
    pointer = ram[start:end, RAM_AREA_POINTER]

    # Every area load passes through the "entering" state, so those frames cut the
    # episode into one run per area visit.
    playing = np.flatnonzero((state != ENTERING) & (state != 0))
    runs = np.split(playing, np.flatnonzero(np.diff(playing) > 1) + 1)
    runs = [run for run in runs if (len(run) > 0)]

    if (len(runs) > 0 and pointer[runs[0][0]] == INTRO_POINTER):
        runs = runs[1:]

    if (len(runs) == 0):
        return np.zeros(0, dtype=np.int64)

    main = [runs[0]]
    main_type = area_type[runs[0][0]]

    # A pipe out of a bonus room comes back to the main area further along, so a later
    # run of the same kind that does not start at the left edge is the main area again.
    # 8-4 is a maze of castle areas that all look like that, so only its first room is kept.
    maze = (episode["world"] == 7 and episode["stage"] == 3)

    for run in runs[1:]:
        if (not maze and area_type[run[0]] == main_type and scroll[run[0]] > 0):
            main.append(run)

    return np.concatenate(main) + start


def true_scroll(ram, index):
    # The stored RAM has every byte 10 turned into 13 (see the note in CLAUDE.md), so
    # the scroll page counts 8, 9, 13, 11. A stretch of 13s is really page 13 only if
    # page 12 or 14 is next to it.
    page = ram[index, RAM_SCROLL_PAGE].astype(np.int64)
    edges = np.flatnonzero(np.diff(index) > 1) + 1
    start = 0

    for end in list(edges) + [len(index)]:
        run = page[start:end]
        thirteen = np.flatnonzero(run == 13)

        for stretch in np.split(thirteen, np.flatnonzero(np.diff(thirteen) > 1) + 1):
            if (len(stretch) == 0):
                continue

            before = run[stretch[0] - 1] if (stretch[0] > 0) else -1
            after = run[stretch[-1] + 1] if (stretch[-1] + 1 < len(run)) else -1

            if (before not in (12, 14) and after not in (12, 14)):
                run[stretch] = 10

        start = end

    return page * 256 + ram[index, RAM_SCROLL_X]


def true_page(page, scroll):
    # The same repair for the page of anything near the screen: Mario, an enemy.
    return np.where((page == 13) & (scroll < 12 * 256), 10, page)


def player_x(ram, index, scroll):
    return true_page(ram[index, RAM_X_PAGE].astype(np.int64), scroll) * 256 + ram[index, RAM_X_ON_PAGE]


def vote(counts):
    # counts shape: (..., classes) -> the most common class
    return counts.argmax(axis=-1).astype(np.uint8)


def build_level(data, ram, episodes):
    frames = data["frames"]

    chosen = [level_frames(ram, episode) for episode in episodes]
    index = np.concatenate(chosen)

    scroll = true_scroll(ram, index)
    width = int(scroll.max()) + FRAME_W
    width += (-width) % TILE

    counts = np.zeros((FRAME_H, width, NUM_COLORS), dtype=np.uint16)
    flat = counts.reshape(-1)
    hud_counts = np.zeros((TILE_TOP, FRAME_W, NUM_COLORS), dtype=np.uint16)

    rows, cols = np.mgrid[0:FRAME_H, 0:FRAME_W]
    base = (rows * width + cols) * NUM_COLORS
    hud_base = (rows[:TILE_TOP] * FRAME_W + cols[:TILE_TOP]) * NUM_COLORS

    stride = max(1, len(index) // VOTE_FRAMES)

    for t, s in zip(index[::stride], scroll[::stride]):
        frame = np.asarray(frames[t]).astype(np.int64)

        # Each pixel votes for its colour at its place in the level. Mario's own box is
        # left out: he stands still at the start and at the flagpole long enough to win.
        keep = np.ones((FRAME_H, FRAME_W), dtype=bool)
        x = int(ram[t, RAM_PLAYER_REL_X])
        y = int(ram[t, RAM_Y_ON_SCREEN]) - CROP_TOP
        keep[max(y - 2, 0):y + 35, max(x - 2, 0):x + 18] = False

        where = base + s * NUM_COLORS + frame
        flat[where[keep]] += 1

        hud_counts.reshape(-1)[hud_base + frame[:TILE_TOP]] += 1

    covered = (counts[TILE_TOP:].sum(axis=(0, 2)) > 0)
    background = vote(counts)
    hud = vote(hud_counts)

    # The backdrop is whatever colour most of the level is. Above the first tile row
    # there is only backdrop and the HUD text, which is drawn separately.
    backdrop = int(np.bincount(background[TILE_TOP:].reshape(-1), minlength=NUM_COLORS).argmax())
    background[:TILE_TOP] = backdrop
    background[:, ~covered] = backdrop

    num_cols = width // TILE
    tile_counts = np.zeros((TILE_ROWS, num_cols, 256), dtype=np.uint16)
    spawns = {}
    lifts = {}

    offset = 0

    for run in chosen:
        run_scroll = scroll[offset:offset + len(run)]
        offset += len(run)

        seen = np.zeros(num_cols, dtype=bool)
        was_active = np.zeros(SLOTS, dtype=bool)
        tracking = {}

        for t, s in zip(run, run_scroll):
            s = int(s)

            # A column is read the first time it is on screen in this run, before Mario
            # has had the chance to break or empty anything in it.
            for col in range(s // TILE, min((s + FRAME_W - 1) // TILE + 1, num_cols)):
                if (seen[col]):
                    continue

                seen[col] = True
                page = RAM_TILES + ((col // 16) % 2) * PAGE_BYTES
                column = ram[t, page + (col % 16):page + PAGE_BYTES:16]
                tile_counts[np.arange(TILE_ROWS), col, column] += 1

            for slot in range(SLOTS - 1):
                active = (ram[t, RAM_ENEMY_ACTIVE + slot] == 1)
                kind = int(ram[t, RAM_ENEMY_TYPE + slot])
                ex = int(true_page(int(ram[t, RAM_ENEMY_X_PAGE + slot]), s)) * 256 + int(ram[t, RAM_ENEMY_X + slot])
                ey = int(ram[t, RAM_ENEMY_Y + slot]) - CROP_TOP

                if (active and not was_active[slot]):
                    tracking.pop(slot, None)

                    # Enemies appear at the right edge exactly where the level data put
                    # them, so the first sighting is the spawn point.
                    if (kind in WALKERS and ex > s + FRAME_W - 2 * TILE):
                        spawns[(kind, (ex + 8) // TILE, ey)] = (kind, ex, ey)

                    if (kind in LIFTS):
                        tracking[slot] = (kind, [])

                if (active and slot in tracking and tracking[slot][0] == kind):
                    on_screen = (ram[t, RAM_ENEMY_Y_SCREEN + slot] == 1)
                    tracking[slot][1].append((ex, ey if (on_screen) else FRAME_H + TILE))

                if (not active and slot in tracking):
                    kind, path = tracking.pop(slot)
                    key = (kind, (path[0][0] + 8) // TILE)

                    # The longest sighting of each platform is kept; its recorded path
                    # is replayed in a loop by the engine.
                    if (len(path) > len(lifts.get(key, (0, []))[1])):
                        lifts[key] = (kind, path)

                was_active[slot] = active

            # A run that ends with platforms still on screen keeps them too.
            if (t == run[-1]):
                for slot, (kind, path) in tracking.items():
                    key = (kind, (path[0][0] + 8) // TILE)
                    if (len(path) > len(lifts.get(key, (0, []))[1])):
                        lifts[key] = (kind, path)

    tiles = vote(tile_counts)
    tiles[:, ~covered.reshape(num_cols, TILE).any(axis=1)] = 0

    # A stretch every player skipped through a pipe has no frames. The level is kept as
    # the pieces that were seen: (first column, one past the last, where Mario comes in,
    # how far he can go). The engine jumps from the end of one piece to the start of
    # the next, the way the pipe did.
    x = player_x(ram, index, scroll)
    y = ram[index, RAM_Y_ON_SCREEN].astype(np.int64) - CROP_TOP
    run_starts = np.concatenate([[0], np.flatnonzero(np.diff(index) > 1) + 1])

    change = np.diff(np.concatenate([[0], covered.astype(np.int8), [0]]))
    bounds = np.stack([np.flatnonzero(change == 1), np.flatnonzero(change == -1)], axis=1)
    pieces = []

    for left, right in bounds:
        if (len(pieces) > 0 and left - pieces[-1][1] < GAP):
            pieces[-1][1] = int(right)
            continue

        pieces.append([int(left), int(right), 0, 0, 0])

    for piece in pieces:
        inside = (x >= piece[0]) & (x < piece[1])
        entries = [i for i in run_starts if (inside[i])]

        if (inside.sum() == 0 or len(entries) == 0):
            piece[4] = -1
            continue

        piece[2], piece[3] = int(x[entries[0]]), int(y[entries[0]])
        piece[4] = int(x[inside].max())

    pieces = np.array([piece for piece in pieces if (piece[4] >= 0)], dtype=np.int64).reshape(-1, 5)

    lift_kinds = np.array([kind for kind, path in lifts.values()], dtype=np.int16)
    lift_lengths = np.array([len(path) for kind, path in lifts.values()], dtype=np.int64)
    lift_paths = np.array([point for kind, path in lifts.values() for point in path], dtype=np.int16).reshape(-1, 2)

    level = {
        "background": background,  # (224, width) NES colour numbers, sprites voted away
        "hud": hud,                # (24, 256) the top of the screen, fixed in place
        "tiles": tiles,            # (13, width / 16) the game's own solid-tile numbers
        "spawns": np.array(sorted(spawns.values(), key=lambda spawn: spawn[1]), dtype=np.int16).reshape(-1, 3),
        "lift_kinds": lift_kinds,
        "lift_lengths": lift_lengths,
        "lift_paths": lift_paths,
        "pieces": pieces,
        "backdrop": np.array(backdrop),
        "area_type": np.array(int(ram[index[0], RAM_AREA_TYPE]))
    }

    return level, index, scroll


def cut(frames, background, t, s, x, y, window, flip):
    # The window around a sprite, with every pixel that matches the level's background
    # marked see-through. x, y is the top left of the window on the screen.
    H, W = window
    out = np.full((H, W), EMPTY, dtype=np.uint8)

    y0, y1 = max(y, 0), min(y + H, FRAME_H)
    x0, x1 = max(x, 0), min(x + W, FRAME_W)

    if (y1 <= y0 or x1 <= x0):
        return None

    frame = np.asarray(frames[t])[y0:y1, x0:x1]
    behind = background[y0:y1, s + x0:s + x1]
    out[y0 - y:y1 - y, x0 - x:x1 - x] = np.where(frame == behind, EMPTY, frame)

    return out[:, ::-1] if (flip) else out


def vote_sprite(cuts):
    # cuts: list of (H, W) -> (H, W), each pixel the colour most cut-outs agree on
    stack = np.stack(cuts)
    counts = (stack[..., None] == np.arange(NUM_COLORS + 1)).sum(axis=0)

    return vote(counts)


def mario_sprites(data, ram, levels, rng):
    # One drawing per (size, graphics offset): the game's own id for which of Mario's
    # poses is on screen. The engine looks the poses up by these ids; POSES in
    # simul/engine.py names them.
    frames = data["frames"]
    sprites = {}
    usage = {}

    index = np.concatenate([index for level, index, scroll in levels.values()])
    scroll = np.concatenate([scroll for level, index, scroll in levels.values()])
    level_of = np.concatenate([np.full(len(index), key) for key, (level, index, scroll) in levels.items()])

    r = ram[index]
    size = np.where(r[:, RAM_SMALL] == 1, 0, np.where(r[:, RAM_POWERUP] == 2, 2, 1))

    # Flashing after a hit, the star, and being half off the screen all give cut-outs
    # that are not the plain drawing.
    clean = (
        np.isin(r[:, RAM_PLAYER_STATE], [0x08, FLAGPOLE, 0x0B])
        & (r[:, RAM_Y_SCREEN] == 1)
        & (r[:, RAM_Y_ON_SCREEN] > 40) & (r[:, RAM_Y_ON_SCREEN] < 200)
        & (r[:, RAM_INJURY_TIMER] == 0) & (r[:, RAM_STAR_TIMER] == 0)
        & ((r[:, RAM_SMALL] == 1) == (r[:, RAM_POWERUP] == 0))
    )

    keys = size.astype(np.int64) * 256 + r[:, RAM_GFX_OFFSET]

    for key in np.unique(keys[clean]):
        members = np.flatnonzero(clean & (keys == key))

        if (len(members) < 20):
            continue

        picked = rng.choice(members, size=min(SAMPLES, len(members)), replace=False)
        cuts = []

        for i in picked:
            level = levels[level_of[i]][0]
            x = int(r[i, RAM_PLAYER_REL_X]) - 8
            y = int(r[i, RAM_Y_ON_SCREEN]) - CROP_TOP - 8
            piece = cut(frames, level["background"], index[i], scroll[i], x, y, MARIO_WINDOW, r[i, RAM_FACING] == 2)

            if (piece is not None):
                cuts.append(piece)

        sprites[int(key)] = vote_sprite(cuts)

        # What Mario was doing while each drawing was up, so the poses can be named.
        m = r[members]
        usage[int(key)] = {
            "count": len(members),
            "air": float((m[:, RAM_FLOAT_STATE] != 0).mean()),
            "still": float((m[:, RAM_X_SPEED] == 0).mean()),
            "skid": float((m[:, RAM_MOVING_DIR] != m[:, RAM_FACING]).mean()),
            "crouch": float((m[:, RAM_CROUCHING] != 0).mean()),
            "flag": float((m[:, RAM_PLAYER_STATE] == FLAGPOLE).mean()),
            "dying": float((m[:, RAM_PLAYER_STATE] == 0x0B).mean()),
            "water": float((m[:, RAM_AREA_TYPE] == WATER).mean())
        }

    return sprites, usage


def object_sprites(data, ram, levels, rng):
    # Enemies, platforms and power-ups, one drawing per (kind, area type, animation
    # frame). Their colours change with the area: goombas are blue underground.
    frames = data["frames"]
    candidates = {}

    for key, (level, index, scroll) in levels.items():
        r = ram[index]

        for slot in range(SLOTS):
            kind = r[:, RAM_ENEMY_TYPE + slot].astype(np.int64)
            ex = true_page(r[:, RAM_ENEMY_X_PAGE + slot].astype(np.int64), scroll) * 256 + r[:, RAM_ENEMY_X + slot] - scroll
            ey = r[:, RAM_ENEMY_Y + slot].astype(np.int64) - CROP_TOP

            # A power-up counts its own states while it rises out of the block, so the
            # "walking normally" test is only for enemies.
            normal = (r[:, RAM_ENEMY_STATE + slot] == 0) | (slot == SLOTS - 1)

            ok = (
                (r[:, RAM_ENEMY_ACTIVE + slot] == 1) & normal
                & (r[:, RAM_ENEMY_Y_SCREEN + slot] == 1)
                & (ex > 16) & (ex < FRAME_W - 64) & (ey > 40) & (ey < 190)
            )

            if (slot == SLOTS - 1):
                ok &= (kind == POWERUP_OBJECT)
                kind = 0x100 + r[:, RAM_POWERUP_TYPE].astype(np.int64)
            else:
                ok &= np.isin(kind, WALKERS + LIFTS)

            # The walk cycle of every enemy flips on the same bit of the frame counter.
            phase = (r[:, RAM_FRAME_COUNTER] >> 3) & 1
            flip = (r[:, RAM_ENEMY_DIR + slot] == 1)

            for i in np.flatnonzero(ok):
                sprite_key = (int(kind[i]), int(level["area_type"]), int(phase[i]))
                candidates.setdefault(sprite_key, []).append((key, index[i], scroll[i], int(ex[i]), int(ey[i]), bool(flip[i])))

    sprites = {}

    for sprite_key, found in candidates.items():
        if (len(found) < 10):
            continue

        cuts = []

        for i in rng.choice(len(found), size=min(SAMPLES, len(found)), replace=False):
            key, t, s, ex, ey, flip = found[i]
            piece = cut(frames, levels[key][0]["background"], t, s, ex - 8, ey - 16, ENEMY_WINDOW, flip)

            if (piece is not None):
                cuts.append(piece)

        sprites[sprite_key] = vote_sprite(cuts)

    return sprites


def block_images(data, ram, levels, rng):
    # What each kind of tile looks like in each level, voted over the places it was seen.
    # The engine needs the used block, to draw over a question block once it is hit.
    frames = data["frames"]
    images = {}

    for key, (level, index, scroll) in levels.items():
        found = {}

        for i in rng.choice(len(index), size=min(400, len(index)), replace=False):
            t, s = index[i], int(scroll[i])

            for col in range(s // TILE + 1, (s + FRAME_W) // TILE - 1):
                page = RAM_TILES + ((col // 16) % 2) * PAGE_BYTES
                column = ram[t, page + (col % 16):page + PAGE_BYTES:16]

                for row in np.flatnonzero(column):
                    y = TILE_TOP + row * TILE
                    x = col * TILE - s

                    if (y + TILE <= FRAME_H):
                        found.setdefault(int(column[row]), []).append(np.asarray(frames[t])[y:y + TILE, x:x + TILE])

        for tile, patches in found.items():
            stack = np.stack(patches[:60])
            counts = (stack[..., None] == np.arange(NUM_COLORS)).sum(axis=0)
            images[(key, tile)] = vote(counts)

    return images


def digit_glyphs(data, ram, levels):
    # The ten digits of the HUD font, as masks, cut from the timer.
    frames = data["frames"]
    level, index, scroll = levels[0]
    counts = np.zeros((10, 8, 8, 2), dtype=np.int64)

    for t in index[::5]:
        frame = np.asarray(frames[t])

        for place in range(3):
            digit = int(ram[t, RAM_TIMER_DIGITS + place])

            if (digit > 9):
                continue

            patch = frame[DIGIT_Y:DIGIT_Y + 8, TIMER_X + 8 * place:TIMER_X + 8 * place + 8]
            lit = (patch != level["backdrop"]).astype(np.int64)
            counts[digit, np.arange(8)[:, None], np.arange(8)[None, :], lit] += 1

    return vote(counts).astype(bool)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(0)

    data = load()
    ram = np.asarray(data["ram"])
    episodes = data["episodes"]

    levels = {}

    for world in range(8):
        for stage in range(4):
            key = world * 4 + stage
            chosen = episodes[(episodes["world"] == world) & (episodes["stage"] == stage)]

            level, index, scroll = build_level(data, ram, chosen)
            levels[key] = (level, index, scroll)

            np.savez_compressed(f"{OUT_DIR}/level_{key:02d}.npz", **level)
            print(
                f"{world + 1}-{stage + 1}: {level['background'].shape[1]} px wide, pieces {level['pieces'][:, [0, 1, 4]].tolist()}, "
                f"{len(level['spawns'])} enemies, {len(level['lift_kinds'])} platforms, {len(index)} frames"
            )

    mario, usage = mario_sprites(data, ram, levels, rng)
    objects = object_sprites(data, ram, levels, rng)
    blocks = block_images(data, ram, levels, rng)
    digits = digit_glyphs(data, ram, levels)

    for key, stats in sorted(usage.items()):
        print(f"mario size {key // 256} offset {key % 256:#04x}: " + "  ".join(f"{name} {value:.2f}" for name, value in stats.items()))

    np.savez_compressed(
        f"{OUT_DIR}/sprites.npz",
        mario_keys=np.array(sorted(mario)),
        mario=np.stack([mario[key] for key in sorted(mario)]),
        object_keys=np.array(sorted(objects)),
        objects=np.stack([objects[key] for key in sorted(objects)]),
        block_keys=np.array(sorted(blocks)),
        blocks=np.stack([blocks[key] for key in sorted(blocks)]),
        digits=digits,
        palette=data["palette"]
    )

    print(f"\n{len(mario)} Mario drawings, {len(objects)} object drawings, {len(blocks)} block images")

if (__name__ == "__main__"):
    main()
