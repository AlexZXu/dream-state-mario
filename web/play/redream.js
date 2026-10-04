// simul/redream.py and the parts of mmodel/dreamer.py and mmodel/flow.py it uses, for
// the browser. The simulation keeps the game's state and the world model draws every
// frame: each step the simulation's exact frame is noised and the model generates from
// there, so what is on screen is the model's output, with its own errors, but it cannot
// lose Mario or wander into another level the way it does on its own.
import {FRAME_H, FRAME_W, FRAME_SKIP} from './engine.js'

// Euler steps per generated frame. Starting from the guide, 2 steps give the same frames
// as 4 (99.4% of pixels matching the simulation, 93% in Mario's box, either way), and 4
// took 56 ms a frame on an M5, over the 50 ms a 20 Hz game has.
const NUM_STEPS = 2
const TAU_CTX = 0.1 // noise put on the context, inside the range it was trained on
const TAU_GUIDE = 0.6 // how much of the simulation's frame is replaced by noise before the model redraws it
const TAU_HUD = 0.7 // the same for the HUD rows alone: a little more is left to the model there
const STILL_TAU = 0.4 // the most noise used while Mario stands still or has barely started moving
const SLOW = 1.0 // pixels per NES frame under which he counts as that; walking pace is 1.56

const LATENT_H = 28
const LATENT_W = 32
const HUD_CELLS = 3 // latent rows that hold the HUD: the top 24 pixels of the frame
const X_POS_SCALE = 8192.0

function gaussian(size) {
    // Box-Muller: two uniform numbers make two normal ones.
    const out = new Float32Array(size)

    for (let i = 0; i < size; i += 2) {
        const radius = Math.sqrt(-2 * Math.log(1 - Math.random()))
        const angle = 2 * Math.PI * Math.random()

        out[i] = radius * Math.cos(angle)

        if (i + 1 < size) {
            out[i + 1] = radius * Math.sin(angle)
        }
    }

    return out
}

export class Redream {
    constructor(ort, sessions, game, model) {
        // sessions: {encoder, cond, denoiser, decoder}; model: the "model" entry of assets.json
        this.ort = ort
        this.sessions = sessions
        this.game = game

        this.L = model.context_frames
        this.C = model.latent_channels
        this.actionDim = model.action_dim
        this.latentSize = this.C * LATENT_H * LATENT_W

        // palette as the encoder wants it: colour number -> red, green, blue in [0, 1]
        this.palette = game.sprites.palette.map(color => color.map(value => value / 255))

        // Which latent values belong to the HUD rows, in every channel.
        this.hud = new Uint8Array(this.latentSize)

        for (let c = 0; c < this.C; c++) {
            this.hud.fill(1, c * LATENT_H * LATENT_W, c * LATENT_H * LATENT_W + HUD_CELLS * LATENT_W)
        }

        this.context = [] // L latents, oldest first
        this.actions = [] // the buttons that led into each context frame
        this.waiting = true

        this.ms = {encode: 0, denoise: 0, decode: 0} // time spent in each network, for the self-test
    }

    powerup() {
        return this.game.fire ? 2 : (this.game.big ? 1 : 0)
    }

    async encode(frame) {
        // frame: 224x256 colour numbers -> normalized latent, Float32Array (C * 28 * 32)
        const plane = FRAME_H * FRAME_W
        const rgb = new Float32Array(3 * plane)

        for (let i = 0; i < plane; i++) {
            const color = this.palette[frame[i]]

            rgb[i] = color[0]
            rgb[plane + i] = color[1]
            rgb[2 * plane + i] = color[2]
        }

        const result = await this.sessions.encoder.run({rgb: new this.ort.Tensor('float32', rgb, [1, 3, FRAME_H, FRAME_W])})

        return result.output.data
    }

    async decode(z) {
        // normalized latent -> 224x256 colour numbers
        const result = await this.sessions.decoder.run({z: new this.ort.Tensor('float32', z, [1, this.C, LATENT_H, LATENT_W])})

        return Uint8Array.from(result.output.data)
    }

    async seed(frame) {
        // Start from this frame: the context is the one frame repeated with no buttons
        // held, which is what standing still at the start of a level looks like.
        const z = await this.encode(frame)

        this.context = Array.from({length: this.L}, () => z)
        this.actions = Array.from({length: this.L}, () => new Float32Array(this.actionDim))
        this.waiting = true
    }

    async sample(guide, held, tauGuide, tauHud) {
        // Rectified flow: tau = 0 is a clean latent and tau = 1 is pure noise, and
        // sampling walks from noise to clean with Euler steps. Here the walk starts
        // part of the way along, from the guide noised to tauGuide.
        const ort = this.ort
        const size = this.latentSize

        // The context is noised a little, as in training, so the model does not trust
        // its own previous frames too literally.
        const context = new Float32Array(this.L * size)

        this.context.forEach((z, i) => {
            const eps = gaussian(size)

            for (let j = 0; j < size; j++) {
                context[i * size + j] = (1 - TAU_CTX) * z[j] + TAU_CTX * eps[j]
            }
        })

        const actions = new Float32Array((this.L + 1) * this.actionDim)
        this.actions.forEach((action, i) => actions.set(action, i * this.actionDim))
        actions.set(held, this.L * this.actionDim)

        // The HUD rows start from more noise than the rest. That costs one extra step,
        // from tauHud down to tauGuide, during which the other cells are held on the
        // guide's path; from there every cell gets the same NUM_STEPS.
        const loose = (tauHud > tauGuide)
        const taus = Array.from({length: NUM_STEPS + 1}, (_, i) => tauGuide * (1 - i / NUM_STEPS))

        if (loose) {
            taus.unshift(tauHud)
        }

        const eps = gaussian(size)
        let z = new Float32Array(size)

        for (let j = 0; j < size; j++) {
            z[j] = (1 - taus[0]) * guide[j] + taus[0] * eps[j]
        }

        const feeds = {
            tau_ctx: new ort.Tensor('float32', new Float32Array([TAU_CTX]), [1]),
            actions: new ort.Tensor('float32', actions, [1, this.L + 1, this.actionDim]),
            level: new ort.Tensor('int64', new BigInt64Array([BigInt(this.game.level)]), [1]),
            x_pos: new ort.Tensor('float32', new Float32Array([this.game.x / X_POS_SCALE]), [1]),
            powerup: new ort.Tensor('int64', new BigInt64Array([BigInt(this.powerup())]), [1]),
            state_mask: new ort.Tensor('float32', new Float32Array([1.0]), [1])
        }
        const contextTensor = new ort.Tensor('float32', context, [1, this.L, this.C, LATENT_H, LATENT_W])

        for (let i = 0; i < taus.length - 1; i++) {
            feeds.tau = new ort.Tensor('float32', new Float32Array([taus[i]]), [1])

            const cond = await this.sessions.cond.run(feeds)
            const result = await this.sessions.denoiser.run({
                z_noisy: new ort.Tensor('float32', z, [1, this.C, LATENT_H, LATENT_W]),
                context: contextTensor,
                cond: cond.output
            })

            const velocity = result.output.data
            const next = new Float32Array(size)
            const hold = (loose && taus[i + 1] >= tauGuide)

            for (let j = 0; j < size; j++) {
                if (hold && !this.hud[j]) {
                    next[j] = (1 - taus[i + 1]) * guide[j] + taus[i + 1] * eps[j]
                } else {
                    next[j] = z[j] - (taus[i] - taus[i + 1]) * velocity[j]
                }
            }

            z = next
        }

        return z
    }

    async step(action) {
        // action: 6 buttons, held for the whole step. Returns the frame to show.
        const game = this.game
        const frame = game.step(action)

        // A respawn or a new level is a new scene: the model is started again from it.
        if (game.cut) {
            await this.seed(frame)
        }

        // Until the player first touches a button in a scene, the model is not used at
        // all. It erases a Mario who stands still (the recordings it learned from
        // almost never show it), and the opening of a level is exactly that. The scene
        // goes on moving meanwhile, so the model's memory is kept on the latest frame.
        if (this.waiting && !action.some(Boolean)) {
            await this.seed(frame)

            return frame
        }

        this.waiting = false

        // The same weakness later on, whenever Mario stops, and for the first steps of
        // setting off again: for those the model is given the simulation's past frames
        // instead of its own, and less noise, so it has no chance to forget him.
        const still = (game.mode === 'play' && Math.abs(game.vx) < SLOW)
        const tauGuide = still ? Math.min(TAU_GUIDE, STILL_TAU) : TAU_GUIDE

        // The model was trained on three button readings per step; the page reads the
        // keyboard once and repeats it.
        const held = new Float32Array(this.actionDim)

        for (let i = 0; i < FRAME_SKIP; i++) {
            held.set(action, i * action.length)
        }

        const start = performance.now()
        const guide = await this.encode(frame)
        const encoded = performance.now()
        const z = await this.sample(guide, held, tauGuide, TAU_HUD)
        const sampled = performance.now()

        this.context = [...this.context.slice(1), still ? guide : z]
        this.actions = [...this.actions.slice(1), held]

        const shown = await this.decode(z)

        this.ms.encode += encoded - start
        this.ms.denoise += sampled - encoded
        this.ms.decode += performance.now() - sampled

        return shown
    }
}
