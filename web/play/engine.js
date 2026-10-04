// simul/engine.py in JavaScript, line for line. test/test_engine.mjs holds it to the
// Python version: the same buttons must give the same frames, pixel for pixel, so any
// change to the rules has to be made in both files.

export const FRAME_H = 224
export const FRAME_W = 256
export const FRAME_SKIP = 3 // NES frames per step: the console runs at 60 fps, the model at 20
export const NUM_LEVELS = 32
const START_LIVES = 3

// order of the 6 entries in an action
export const LEFT = 0
export const RIGHT = 1
export const UP = 2
export const DOWN = 3
export const A = 4
export const B = 5

// Mario's movement, in pixels per NES frame. These are the original game's numbers.
const WALK_MAX = 1.5625
const RUN_MAX = 2.5625
const WALK_ACCEL = 0.0371
const RUN_ACCEL = 0.0557
const RELEASE = 0.0508 // slowing down with no direction held
const SKID = 0.1016 // slowing down with the opposite direction held
const JUMP_SPEED = 4.0
const RUNNING_JUMP_SPEED = 5.0
const RUNNING_JUMP_AT = 2.3125
const HOLD_GRAVITY = [0.125, 0.1172, 0.1563] // while A is held on the way up; by speed at take-off
const FALL_GRAVITY = [0.4375, 0.375, 0.5625]
const MAX_FALL = 4.0
const STOMP_BOUNCE = 4.0

const SWIM_STROKE = 1.5
const SWIM_GRAVITY = 0.04
const SWIM_MAX_FALL = 1.5
const SWIM_MAX = 1.0

const ENEMY_SPEED = 0.5
const ENEMY_GRAVITY = 0.3
const MUSHROOM_SPEED = 1.0

const SCROLL_AT = 112 // Mario's screen x past which the camera follows him; it never goes back
const BOX_LEFT = 3 // Mario's hitbox inside his 16x32 box
const BOX_RIGHT = 13
const SMALL_TOP = 19
const BIG_TOP = 7

const TIMER_START = 400
const TIMER_FRAMES = 24 // NES frames per tick of the clock
const HURT_FRAMES = 120 // flashing and untouchable after a hit
const DYING_FRAMES = 150
const CLEAR_FRAMES = 40
const SQUASH_FRAMES = 30

const TILE = 16
const TILE_ROWS = 13
const TILE_TOP = 24 // screen row of the first tile row
const EMPTY = 64 // the "see-through" colour number in a sprite
const ENEMY_WINDOW_W = 64
const DIGIT_Y = 16
const TIMER_X = 208
const WATER = 0

// Tile numbers, the game's own. Everything else that is not 0 is solid.
const BRICKS = [0x51, 0x52]
const POWERUP_BLOCKS = [0xC1, 0x55, 0x5A]
const ITEM_BLOCKS = [0xC0, 0xC1, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x5B, 0x5C, 0x5D, 0x5E]
const HIDDEN = [0x5F, 0x60] // solid only from below
const COINS = [0xC2, 0xC3]
const FLAGPOLE = [0x24, 0x25]
const VINE = 0x26
const USED = 0xC4
const PASSABLE = [0, ...HIDDEN, ...COINS, ...FLAGPOLE, VINE]
const REDRAWN = [...BRICKS, ...ITEM_BLOCKS, ...COINS]

// Which drawing of Mario goes with which pose: the game's graphics offsets. Small Mario
// has his own set; big and fire share one.
const POSES = {
    small: {walk: [0x60, 0x68, 0x70], skid: 0x78, jump: 0x80, swim: [0x88, 0x90, 0x98], climb: 0xA0, die: 0xB0, stand: 0xB8, crouch: 0xB8},
    big: {walk: [0x00, 0x08, 0x10], skid: 0x18, jump: 0x20, swim: [0x28, 0x30, 0x38], climb: 0x40, die: 0x20, stand: 0xC8, crouch: 0x50}
}

const GOOMBA = 0x06
const SAME_LOOK = {0x01: 0x03, 0x04: 0x00} // koopa kinds that only differ in how they walk
const MUSHROOM = 0x100
const FLOWER = 0x101

function clip(value, low, high) {
    return Math.min(Math.max(value, low), high)
}

function images(blob, entry) {
    // An (N, H, W) array of the asset block -> N sprites.
    const [count, height, width] = entry.shape
    const out = []

    for (let i = 0; i < count; i++) {
        const start = entry.offset + i * height * width
        out.push({data: blob.subarray(start, start + height * width), height: height, width: width})
    }

    return out
}

export function makeSprites(index, blob) {
    // index is assets.json and blob the unzipped assets.bin, see export/export_web.py
    const mario = new Map()
    const objects = new Map()
    const blocks = new Map()

    images(blob, index.mario).forEach((image, i) => mario.set(index.mario_keys[i], image))
    images(blob, index.objects).forEach((image, i) => objects.set(index.object_keys[i].join(','), image))
    images(blob, index.blocks).forEach((image, i) => blocks.set(index.block_keys[i].join(','), image))

    return {
        mario: mario,
        objects: objects,
        blocks: blocks,
        blockKeys: index.block_keys,
        digits: images(blob, index.digits),
        palette: index.palette
    }
}

function findObject(sprites, kind, areaType, phase) {
    // Not every enemy was seen in every kind of area; the same enemy from another
    // area has the wrong colours, which still beats a missing one.
    kind = (kind in SAME_LOOK) ? SAME_LOOK[kind] : kind

    for (const area of [areaType, 1, 2, 3, 0]) {
        const sprite = sprites.objects.get(`${kind},${area},${phase}`)

        if (sprite !== undefined) {
            return sprite
        }
    }

    return null
}

function findBlock(sprites, level, tile) {
    const own = sprites.blocks.get(`${level},${tile}`)

    if (own !== undefined) {
        return own
    }

    for (const [other, kind] of sprites.blockKeys) {
        if (kind === tile) {
            return sprites.blocks.get(`${other},${kind}`)
        }
    }

    return null
}

function extent(sprite) {
    // [top, bottom, left, right] of the drawn pixels inside a sprite's window
    let top = sprite.height, bottom = 0, left = sprite.width, right = 0

    for (let row = 0; row < sprite.height; row++) {
        for (let col = 0; col < sprite.width; col++) {
            if (sprite.data[row * sprite.width + col] !== EMPTY) {
                top = Math.min(top, row)
                bottom = Math.max(bottom, row + 1)
                left = Math.min(left, col)
                right = Math.max(right, col + 1)
            }
        }
    }

    return [top, bottom, left, right]
}

export class Game {
    // Super Mario Bros rebuilt from the recorded frames. step() takes the buttons and
    // returns the next frame; nothing here is generated, this is ordinary game code.

    constructor(index, blob, level = 0) {
        this.index = index
        this.blob = blob
        this.sprites = makeSprites(index, blob)

        this.lives = START_LIVES
        this.score = 0
        this.coins = 0
        this.big = false
        this.fire = false

        this.load(level)
    }

    load(level, piece = 0) {
        const data = this.index.levels[level]
        const [height, width] = data.background.shape

        this.level = level
        this.width = width
        this.cols = data.tiles.shape[1]
        this.background = this.blob.slice(data.background.offset, data.background.offset + height * width)
        this.tiles = this.blob.slice(data.tiles.offset, data.tiles.offset + TILE_ROWS * this.cols)
        this.hud = this.blob.subarray(data.hud.offset, data.hud.offset + TILE_TOP * FRAME_W)
        this.pieces = data.pieces
        this.backdrop = data.backdrop
        this.areaType = data.area_type
        this.water = (this.areaType === WATER)

        // Different runs first saw the same enemy a pixel or two apart, so sightings of
        // one kind at one height within a few pixels of each other are one enemy.
        this.spawns = []

        for (const [kind, x, y] of data.spawns) {
            if (!this.spawns.some(([other, place, height]) => kind === other && y === height && Math.abs(x - place) <= 6)) {
                this.spawns.push([kind, x, y])
            }
        }

        this.spawned = this.spawns.map(() => false)

        this.liftKinds = data.lift_kinds
        this.liftPaths = data.lift_paths
        this.liftStarted = this.liftKinds.map(() => false)

        // The HUD text is the lightest thing on the top rows.
        const counts = new Array(EMPTY).fill(0)
        let any = false

        for (const color of this.hud) {
            if (color !== this.backdrop) {
                counts[color] += 1
                any = true
            }
        }

        this.textColor = any ? counts.indexOf(Math.max(...counts)) : 0x30

        // The stitched background shows each block the way most recordings left it:
        // question blocks used, bricks broken, coins taken. The tile grid was read
        // before anyone touched them, so they are drawn back in from it.
        for (let row = 0; row < TILE_ROWS; row++) {
            for (let col = 0; col < this.cols; col++) {
                if (REDRAWN.includes(this.tiles[row * this.cols + col])) {
                    this.setTile(col, row, this.tiles[row * this.cols + col])
                }
            }
        }

        this.timer = TIMER_START
        this.frame = 0
        this.enter(piece)
    }

    enter(piece) {
        this.piece = piece
        const [left, right, x, y, reach] = this.pieces[piece]

        this.x = x
        this.y = y
        this.vx = 0.0
        this.vy = 0.0
        this.facing = 1
        this.grounded = false
        this.gravityClass = 0
        this.crouching = false
        this.walkDistance = 0.0
        this.riding = null

        this.scroll = Math.trunc(clip(x - SCROLL_AT, left, Math.max(right - FRAME_W, left)))
        this.enemies = []
        this.items = []
        this.lifts = []

        this.mode = 'play'
        this.modeFrames = 0
        this.hurtFrames = 0
        this.jumpHeld = true // a button held through the cut must be pressed again
        this.cut = true // tells the viewer that this frame does not follow from the last
    }

    solid(col, row, fromBelow = false) {
        if (col < 0) {
            return true
        }
        if (row < 0 || row >= TILE_ROWS || col >= this.cols) {
            return false
        }

        const tile = this.tiles[row * this.cols + col]

        if (fromBelow && HIDDEN.includes(tile)) {
            return true
        }

        return !PASSABLE.includes(tile)
    }

    setTile(col, row, tile) {
        this.tiles[row * this.cols + col] = tile
        const image = (tile !== 0) ? findBlock(this.sprites, this.level, tile) : null
        const top = TILE_TOP + row * TILE
        const left = col * TILE

        for (let y = top; y < Math.min(top + TILE, FRAME_H); y++) {
            for (let x = left; x < left + TILE; x++) {
                this.background[y * this.width + x] = (image === null) ? this.backdrop : image.data[(y - top) * TILE + (x - left)]
            }
        }
    }

    bump(col, row) {
        const tile = this.tiles[row * this.cols + col]

        if (BRICKS.includes(tile)) {
            if (this.big) {
                this.setTile(col, row, 0)
                this.score += 50
            }
        } else if (ITEM_BLOCKS.includes(tile) || HIDDEN.includes(tile)) {
            this.setTile(col, row, USED)

            if (POWERUP_BLOCKS.includes(tile)) {
                const kind = this.big ? FLOWER : MUSHROOM
                this.items.push({kind: kind, x: col * TILE, y: TILE_TOP + (row - 1) * TILE - 8, vx: (kind === MUSHROOM) ? MUSHROOM_SPEED : 0.0, vy: 0.0})
            } else {
                this.coins += 1
                this.score += 200
            }
        }

        // Anything standing on the block when it is hit from below is knocked off.
        for (const enemy of this.enemies) {
            if (Math.abs(enemy.x - col * TILE) < 12 && Math.abs(enemy.y + 24 - (TILE_TOP + row * TILE)) < 4) {
                enemy.squashed = SQUASH_FRAMES
                this.score += 100
            }
        }
    }

    top() {
        return (this.big && !this.crouching) ? BIG_TOP : SMALL_TOP
    }

    wall(col, firstRow, lastRow) {
        for (let row = firstRow; row <= lastRow; row++) {
            if (this.solid(col, row)) {
                return true
            }
        }

        return false
    }

    moveX(dx) {
        this.x += dx
        const left = this.x + BOX_LEFT
        const right = this.x + BOX_RIGHT - 0.01

        const firstRow = Math.floor((this.y + this.top() - TILE_TOP) / TILE)
        const lastRow = Math.floor((this.y + 31 - TILE_TOP) / TILE)

        if (dx > 0 && this.wall(Math.floor(right / TILE), firstRow, lastRow)) {
            this.x = Math.floor(right / TILE) * TILE - BOX_RIGHT
            this.vx = 0.0
        }

        if (dx < 0 && this.wall(Math.floor(left / TILE), firstRow, lastRow)) {
            this.x = (Math.floor(left / TILE) + 1) * TILE - BOX_LEFT
            this.vx = 0.0
        }

        // The camera never scrolls back, and the left edge of the screen is a wall.
        if (this.x < this.scroll) {
            this.x = this.scroll
            this.vx = Math.max(this.vx, 0.0)
        }
    }

    moveY(dy) {
        const before = this.y + 32
        this.y += dy
        this.grounded = false

        const cols = [Math.floor((this.x + BOX_LEFT + 1) / TILE), Math.floor((this.x + BOX_RIGHT - 1) / TILE)]

        if (dy >= 0) {
            const row = Math.floor((this.y + 32 - TILE_TOP) / TILE)
            const tileTop = TILE_TOP + row * TILE

            // Only a floor Mario came down onto from above counts, so that being pushed
            // into the side of a wall does not pop him up on top of it.
            if ((this.solid(cols[0], row) || this.solid(cols[1], row)) && before <= tileTop + MAX_FALL + 1) {
                this.y = tileTop - 32
                this.vy = 0.0
                this.grounded = true
            }
        } else {
            const row = Math.floor((this.y + this.top() - TILE_TOP) / TILE)
            const col = Math.floor((this.x + 8) / TILE)

            // The head is tested at its middle, like the game does, so Mario slides past
            // the corner of a block instead of catching on it.
            if (this.solid(col, row, true)) {
                this.y = TILE_TOP + (row + 1) * TILE - this.top()
                this.vy = 1.0
                this.bump(col, row)
            }
        }
    }

    control(action, jumpPressed) {
        let direction = action[RIGHT] - action[LEFT]
        this.crouching = Boolean(action[DOWN]) && this.big && this.grounded

        if (this.crouching) {
            direction = 0
        }

        let topSpeed, accel

        if (this.water) {
            [topSpeed, accel] = [SWIM_MAX, WALK_ACCEL]
        } else {
            [topSpeed, accel] = action[B] ? [RUN_MAX, RUN_ACCEL] : [WALK_MAX, WALK_ACCEL]
        }

        if (direction !== 0) {
            if (this.grounded) {
                this.facing = direction
            }

            // Pushing against the way Mario is moving is a skid, which brakes harder.
            if (this.vx * direction < 0) {
                this.vx += direction * SKID
            } else if (Math.abs(this.vx) < topSpeed) {
                this.vx += direction * accel
            } else if (this.grounded) {
                this.vx -= Math.sign(this.vx) * RELEASE
            }
        } else if (this.grounded || this.water) {
            this.vx -= Math.sign(this.vx) * Math.min(RELEASE, Math.abs(this.vx))
        }

        if (this.water) {
            if (jumpPressed) {
                this.vy = -SWIM_STROKE
            }

            this.vy = Math.min(this.vy + SWIM_GRAVITY, SWIM_MAX_FALL)

            if (this.y < TILE_TOP - 8) {
                this.y = TILE_TOP - 8
                this.vy = Math.max(this.vy, 0.0)
            }

            return
        }

        if (jumpPressed && this.grounded) {
            const speed = Math.abs(this.vx)
            this.vy = -((speed >= RUNNING_JUMP_AT) ? RUNNING_JUMP_SPEED : JUMP_SPEED)
            this.gravityClass = (speed < 1.0) ? 0 : ((speed < RUNNING_JUMP_AT) ? 1 : 2)
            this.grounded = false
            this.riding = null
        }

        // Holding A on the way up is what makes a jump high: gravity is weaker for as
        // long as it is held.
        const rising = (this.vy < 0 && action[A])
        const gravity = rising ? HOLD_GRAVITY[this.gravityClass] : FALL_GRAVITY[this.gravityClass]
        this.vy = Math.min(this.vy + gravity, MAX_FALL)
    }

    touchTiles() {
        // Coins and the flagpole are tiles Mario overlaps instead of bumping into.
        for (const [px, py] of [[this.x + 8, this.y + this.top() + 2], [this.x + 8, this.y + 28]]) {
            const col = Math.floor(px / TILE)
            const row = Math.floor((py - TILE_TOP) / TILE)

            if (row < 0 || row >= TILE_ROWS || col >= this.cols) {
                continue
            }

            const tile = this.tiles[row * this.cols + col]

            if (COINS.includes(tile)) {
                this.setTile(col, row, 0)
                this.coins += 1
                this.score += 200
            }

            if (FLAGPOLE.includes(tile) && this.mode === 'play') {
                this.mode = 'flag'
                this.x = col * TILE - 6
                this.vx = this.vy = 0.0
                this.facing = 1
                this.score += 1000
            }
        }
    }

    spawn() {
        this.spawns.forEach(([kind, x, y], i) => {
            if (!this.spawned[i] && this.scroll <= x && x <= this.scroll + FRAME_W + TILE) {
                this.spawned[i] = true
                this.enemies.push({kind: kind, x: x, y: y, vx: -ENEMY_SPEED, vy: 0.0, squashed: 0})
            }
        })

        this.liftPaths.forEach((path, i) => {
            if (!this.liftStarted[i] && this.scroll - 64 <= path[0][0] && path[0][0] <= this.scroll + FRAME_W + TILE) {
                this.liftStarted[i] = true
                const sprite = findObject(this.sprites, this.liftKinds[i], this.areaType, 0)

                if (sprite !== null) {
                    this.lifts.push({kind: this.liftKinds[i], path: path, t: 0, step: 1, x: path[0][0], y: path[0][1], extent: extent(sprite)})
                }
            }
        })
    }

    walker(thing) {
        // Shared by enemies and the mushroom: walk, fall, turn round at a wall.
        thing.vy = Math.min(thing.vy + ENEMY_GRAVITY, MAX_FALL)
        thing.y += thing.vy

        const row = Math.floor((thing.y + 24 - TILE_TOP) / TILE)

        if (this.solid(Math.floor((thing.x + 4) / TILE), row) || this.solid(Math.floor((thing.x + 12) / TILE), row)) {
            thing.y = TILE_TOP + row * TILE - 24
            thing.vy = 0.0
        }

        thing.x += thing.vx
        const ahead = thing.x + ((thing.vx > 0) ? 14 : 2)

        if (this.solid(Math.floor(ahead / TILE), Math.floor((thing.y + 16 - TILE_TOP) / TILE))) {
            thing.vx = -thing.vx
            thing.x += 2 * thing.vx
        }
    }

    hurt() {
        if (this.hurtFrames > 0) {
            return
        }

        if (this.big) {
            this.big = this.fire = false
            this.hurtFrames = HURT_FRAMES
        } else {
            this.die(true)
        }
    }

    die(hop) {
        this.mode = 'dying'
        this.modeFrames = 0
        this.big = this.fire = false
        this.vx = 0.0
        this.vy = hop ? -JUMP_SPEED : 0.0
    }

    objects() {
        for (const lift of this.lifts) {
            // The platform replays its recorded path, back and forth, and carries
            // whatever stands on it.
            const path = lift.path

            if (path.length > 1) {
                if (!(0 <= lift.t + lift.step && lift.t + lift.step < path.length)) {
                    lift.step = -lift.step
                }
                lift.t += lift.step
            }

            const [x, y] = path[lift.t]
            const dx = x - lift.x
            const dy = y - lift.y
            lift.x = x
            lift.y = y

            const [top, bottom, left, right] = lift.extent
            const surface = y - 16 + top
            const feet = this.y + 32
            const over = (x - 8 + left - BOX_RIGHT < this.x && this.x < x - 8 + right - BOX_LEFT)

            if (this.riding === lift) {
                // A recorded platform can jump (it wraps round the screen); Mario is
                // only carried by movement he could have kept up with.
                if (Math.abs(dx) > 8 || Math.abs(dy) > 8 || !over) {
                    this.riding = null
                } else {
                    this.moveX(dx)
                    this.y = surface - 32
                    this.vy = 0.0
                    this.grounded = true
                }
            } else if (this.mode === 'play' && this.vy >= 0 && surface - 2 <= feet && feet <= surface + 6 + Math.abs(dy) && over) {
                this.riding = lift
                this.y = surface - 32
                this.vy = 0.0
                this.grounded = true
            }
        }

        for (const enemy of this.enemies) {
            if (enemy.squashed > 0) {
                enemy.squashed -= 1
                continue
            }

            this.walker(enemy)

            if (this.mode !== 'play') {
                continue
            }

            const overlap = (Math.abs(enemy.x - this.x) < 12 && this.y + this.top() < enemy.y + 24 && this.y + 32 > enemy.y + 10)

            if (overlap) {
                // Coming down onto an enemy is a stomp; touching it any other way hurts.
                if (this.vy > 0 && this.y + 32 < enemy.y + 22) {
                    enemy.squashed = SQUASH_FRAMES
                    this.vy = -STOMP_BOUNCE
                    this.score += 100
                } else {
                    this.hurt()
                }
            }
        }

        this.enemies = this.enemies.filter(enemy => enemy.y < FRAME_H + 32 && enemy.x > this.scroll - 48 && enemy.squashed !== 1)

        for (const item of this.items) {
            if (item.kind === MUSHROOM) {
                this.walker(item)
            }

            if (this.mode === 'play' && Math.abs(item.x - this.x) < 12 && Math.abs(item.y + 8 - this.y - 16) < 20) {
                item.kind = null
                this.score += 1000

                if (this.big) {
                    this.fire = true
                }
                this.big = true
            }
        }

        this.items = this.items.filter(item => item.kind !== null && item.y < FRAME_H + 32)
    }

    tick(action, jumpPressed) {
        // One NES frame.
        this.frame += 1
        this.modeFrames += 1
        this.hurtFrames = Math.max(this.hurtFrames - 1, 0)

        const [left, right, entryX, entryY, reach] = this.pieces[this.piece]
        const lastPiece = (this.piece === this.pieces.length - 1)

        if (this.mode === 'play') {
            if (this.frame % TIMER_FRAMES === 0) {
                this.timer = Math.max(this.timer - 1, 0)
            }

            this.control(action, jumpPressed)

            if (this.riding === null) {
                this.moveX(this.vx)
                this.moveY(this.vy)
            } else {
                this.moveX(this.vx)
            }

            this.walkDistance += Math.abs(this.vx)
            this.touchTiles()

            if (this.y > FRAME_H + 16) {
                this.die(false)
            } else if (this.mode === 'play' && this.x >= reach - 2) {
                // The end of what the recordings show. In the middle of a level that
                // is a pipe every player took, so the game cuts to where it came out.
                if (lastPiece) {
                    this.mode = 'clear'
                    this.modeFrames = 0
                } else {
                    this.enter(this.piece + 1)
                    return
                }
            }
        } else if (this.mode === 'flag') {
            // Slide down the pole, then walk off to the castle.
            this.y += 2.0
            const row = Math.floor((this.y + 32 - TILE_TOP) / TILE)

            if (this.solid(Math.floor((this.x + 8) / TILE), row) || this.solid(Math.floor((this.x + 24) / TILE), row) || this.y > FRAME_H - 56) {
                this.y = Math.min(TILE_TOP + row * TILE - 32, this.y)
                this.mode = 'walkoff'
                this.x += 12
            }
        } else if (this.mode === 'walkoff') {
            this.vx = WALK_MAX
            this.vy = Math.min(this.vy + FALL_GRAVITY[0], MAX_FALL)
            this.moveX(this.vx)
            this.moveY(this.vy)
            this.walkDistance += this.vx

            if (this.x >= reach - 2 || this.modeFrames > 400) {
                this.mode = 'clear'
                this.modeFrames = 0
            }
        } else if (this.mode === 'clear') {
            if (this.modeFrames >= CLEAR_FRAMES) {
                this.load((this.level + 1) % NUM_LEVELS)
                return
            }
        } else if (this.mode === 'dying') {
            this.vy = Math.min(this.vy + HOLD_GRAVITY[0], MAX_FALL)
            this.y += this.vy

            if (this.modeFrames >= DYING_FRAMES) {
                this.lives -= 1

                if (this.lives === 0) {
                    this.lives = START_LIVES
                    this.score = 0
                    this.coins = 0
                    this.load(0)
                } else {
                    this.load(this.level, this.piece)
                }

                return
            }
        }

        // The camera follows Mario to the right and stops at the end of what was seen.
        this.scroll = Math.trunc(clip(Math.max(this.scroll, this.x - SCROLL_AT), left, Math.max(right - FRAME_W, left)))

        this.spawn()
        this.objects()
    }

    step(action) {
        // action: 6 numbers, 0 or 1, in LEFT RIGHT UP DOWN A B order -> a 224x256 frame
        // of NES colour numbers. The buttons are held for FRAME_SKIP NES frames.
        const jumpPressed = Boolean(action[A]) && !this.jumpHeld
        this.jumpHeld = Boolean(action[A])
        this.cut = false

        for (let i = 0; i < FRAME_SKIP; i++) {
            this.tick(action, jumpPressed && i === 0)
        }

        return this.render()
    }

    pose() {
        const poses = this.big ? POSES.big : POSES.small

        if (this.mode === 'dying') {
            return poses.die
        }
        if (this.mode === 'flag') {
            return poses.climb
        }
        if (this.water && !this.grounded) {
            return poses.swim[Math.floor(this.frame / 8) % 3]
        }
        if (!this.grounded) {
            return poses.jump
        }
        if (this.crouching) {
            return poses.crouch
        }
        if (Math.abs(this.vx) < 0.05) {
            return poses.stand
        }
        if (this.vx * this.facing < 0) {
            return poses.skid
        }

        return poses.walk[Math.trunc(this.walkDistance / 5) % 3]
    }

    draw(frame, sprite, x, y, flip = false) {
        // x, y is the sprite's top left on screen; flip mirrors it left to right
        for (let row = Math.max(y, 0); row < Math.min(y + sprite.height, FRAME_H); row++) {
            for (let col = Math.max(x, 0); col < Math.min(x + sprite.width, FRAME_W); col++) {
                const across = flip ? sprite.width - 1 - (col - x) : col - x
                const color = sprite.data[(row - y) * sprite.width + across]

                if (color !== EMPTY) {
                    frame[row * FRAME_W + col] = color
                }
            }
        }
    }

    number(frame, value, digits, x) {
        const text = String(value).padStart(digits, '0').slice(-digits)

        for (let i = 0; i < digits; i++) {
            const glyph = this.sprites.digits[Number(text[i])].data

            for (let row = 0; row < 8; row++) {
                for (let col = 0; col < 8; col++) {
                    frame[(DIGIT_Y + row) * FRAME_W + x + 8 * i + col] = glyph[row * 8 + col] ? this.textColor : this.backdrop
                }
            }
        }
    }

    squash(sprite) {
        // A stomped enemy is its own drawing pressed flat onto the ground.
        const [top, bottom, left, right] = extent(sprite)
        const flat = new Uint8Array(sprite.data.length).fill(EMPTY)
        const count = Math.floor((bottom - top + 1) / 2)

        for (let i = 0; i < count; i++) {
            const from = (count === 1) ? top : Math.trunc((i === count - 1) ? bottom - 1 : top + i * ((bottom - 1 - top) / (count - 1)))
            const to = bottom - count + i

            flat.set(sprite.data.subarray(from * sprite.width, (from + 1) * sprite.width), to * sprite.width)
        }

        return {data: flat, height: sprite.height, width: sprite.width}
    }

    render() {
        const frame = new Uint8Array(FRAME_H * FRAME_W).fill(this.backdrop)
        const visibleWidth = Math.min(FRAME_W, this.width - this.scroll)

        for (let row = 0; row < FRAME_H; row++) {
            const start = row * this.width + this.scroll
            frame.set(this.background.subarray(start, start + visibleWidth), row * FRAME_W)
        }

        const phase = (this.frame >> 3) & 1

        for (const lift of this.lifts) {
            const sprite = findObject(this.sprites, lift.kind, this.areaType, 0)
            this.draw(frame, sprite, lift.x - 8 - this.scroll, lift.y - 16)
        }

        for (const item of this.items) {
            let sprite = findObject(this.sprites, item.kind, this.areaType, phase)

            if (sprite === null) {
                sprite = findObject(this.sprites, MUSHROOM, this.areaType, 0)
            }
            if (sprite !== null) {
                this.draw(frame, sprite, Math.trunc(item.x) - 8 - this.scroll, Math.trunc(item.y) - 16)
            }
        }

        for (const enemy of this.enemies) {
            let sprite = findObject(this.sprites, enemy.kind, this.areaType, phase)

            if (sprite === null) {
                sprite = findObject(this.sprites, GOOMBA, this.areaType, phase)
            }
            if (sprite === null) {
                continue
            }

            if (enemy.squashed > 0) {
                sprite = this.squash(sprite)
            }

            if (enemy.vx > 0) {
                this.draw(frame, sprite, Math.trunc(enemy.x) + 24 - ENEMY_WINDOW_W - this.scroll, Math.trunc(enemy.y) - 16, true)
            } else {
                this.draw(frame, sprite, Math.trunc(enemy.x) - 8 - this.scroll, Math.trunc(enemy.y) - 16)
            }
        }

        // Mario flashes while he cannot be hurt, and is gone once the level is over.
        const visible = (this.mode !== 'clear' && Math.floor(this.hurtFrames / 4) % 2 === 0)
        const size = this.fire ? 2 : (this.big ? 1 : 0)
        const sprite = this.sprites.mario.get(size * 256 + this.pose())

        if (visible && sprite !== undefined) {
            this.draw(frame, sprite, Math.trunc(this.x) - 8 - this.scroll, Math.trunc(this.y) - 8, this.facing < 0)
        }

        // The HUD sits on top of everything and does not scroll.
        for (let i = 0; i < TILE_TOP * FRAME_W; i++) {
            if (this.hud[i] !== this.backdrop) {
                frame[i] = this.hud[i]
            }
        }

        this.number(frame, this.score, 6, 24)
        this.number(frame, this.coins % 100, 2, 104)
        this.number(frame, this.timer, 3, TIMER_X)

        return frame
    }
}
