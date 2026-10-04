import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs'
import {Game, FRAME_H, FRAME_W, LEFT, RIGHT, UP, DOWN, A, B} from './engine.js'
import {Redream} from './redream.js'

const STEP_MS = 50 // 20 steps per second, the rate the model was trained at

const KEYS = {
    ArrowLeft: LEFT,
    ArrowRight: RIGHT,
    ArrowUp: UP,
    ArrowDown: DOWN,
    KeyX: A, // jump
    KeyZ: B  // run
}

const status = document.getElementById('status')
const canvas = document.getElementById('screen')
const screen = canvas.getContext('2d')
const held = new Set()

function readKeys() {
    const action = [0, 0, 0, 0, 0, 0]

    for (const code of held) {
        action[KEYS[code]] = 1
    }

    // A real d-pad cannot press two opposite directions, so the pair cancels out.
    if (action[LEFT] && action[RIGHT]) {
        action[LEFT] = action[RIGHT] = 0
    }
    if (action[UP] && action[DOWN]) {
        action[UP] = action[DOWN] = 0
    }

    return action
}

function show(frame, palette, image) {
    // frame: 224x256 colour numbers -> the canvas
    for (let i = 0; i < FRAME_H * FRAME_W; i++) {
        const color = palette[frame[i]]

        image.data[4 * i] = color[0]
        image.data[4 * i + 1] = color[1]
        image.data[4 * i + 2] = color[2]
        image.data[4 * i + 3] = 255
    }

    screen.putImageData(image, 0, 0)
}

async function fetchBytes(path, label) {
    // The models are 60 MB between them, so the download says how far along it is.
    const response = await fetch(path)
    const total = Number(response.headers.get('Content-Length'))
    const reader = response.body.getReader()
    const chunks = []
    let received = 0

    while (true) {
        const {done, value} = await reader.read()

        if (done) {
            break
        }

        chunks.push(value)
        received += value.length
        status.textContent = total ? `loading ${label}: ${Math.round(100 * received / total)}%` : `loading ${label}`
    }

    return new Uint8Array(await new Blob(chunks).arrayBuffer())
}

async function load() {
    const index = await (await fetch('assets.json')).json()

    // assets.bin is gzipped by export_web.py, whatever the server does on top.
    const zipped = await fetchBytes('assets.bin', 'levels')
    const stream = new Blob([zipped]).stream().pipeThrough(new DecompressionStream('gzip'))
    const blob = new Uint8Array(await new Response(stream).arrayBuffer())

    // The conditioning network is a few small matrix multiplies and has to stay in
    // float32, so it runs on the CPU; the rest run on the GPU.
    const sessions = {
        cond: await ort.InferenceSession.create(await fetchBytes('cond.onnx', 'model 1 of 4'), {executionProviders: ['wasm']}),
        encoder: await ort.InferenceSession.create(await fetchBytes('encoder.onnx', 'model 2 of 4'), {executionProviders: ['webgpu']}),
        decoder: await ort.InferenceSession.create(await fetchBytes('decoder.onnx', 'model 3 of 4'), {executionProviders: ['webgpu']}),
        denoiser: await ort.InferenceSession.create(await fetchBytes('denoiser.onnx', 'model 4 of 4'), {executionProviders: ['webgpu']})
    }

    const game = new Game(index, blob)
    const redream = new Redream(ort, sessions, game, index.model)

    return {game, redream}
}

async function selfTest(game, redream, steps) {
    // ?selftest=200 plays that many steps with fixed buttons as fast as it can and
    // reports the time per frame and how much of each frame matches the simulation's.
    // Headless Chrome reads the result off the page.
    const times = []
    const matches = []
    let shown = game.render()
    await redream.seed(shown)

    for (let step = 0; step < steps; step++) {
        const action = [0, 1, 0, 0, (step % 30 < 10) ? 1 : 0, 1]
        const start = performance.now()

        shown = await redream.step(action)
        times.push(performance.now() - start)

        const exact = game.render()
        let same = 0

        for (let i = 0; i < FRAME_H * FRAME_W; i++) {
            same += (shown[i] === exact[i])
        }

        matches.push(same / (FRAME_H * FRAME_W))
    }

    show(shown, game.sprites.palette, screen.createImageData(FRAME_W, FRAME_H))

    times.sort((a, b) => a - b)
    const average = matches.reduce((a, b) => a + b, 0) / matches.length

    status.textContent =
        `selftest: ${steps} steps, median ${times[Math.floor(steps / 2)].toFixed(1)} ms per frame, ` +
        `worst ${times[steps - 1].toFixed(1)} ms, ${(100 * average).toFixed(2)}% of pixels match the simulation, ` +
        `lowest frame ${(100 * Math.min(...matches)).toFixed(2)}%; ` +
        `per frame: encode ${(redream.ms.encode / steps).toFixed(1)} ms, denoise ${(redream.ms.denoise / steps).toFixed(1)} ms, decode ${(redream.ms.decode / steps).toFixed(1)} ms`
    document.title = 'selftest done'
}

async function main() {
    if (!navigator.gpu) {
        status.textContent = 'This needs WebGPU, which this browser does not have. Try a current Chrome, Edge or Safari.'
        return
    }

    const {game, redream} = await load()
    const steps = Number(new URLSearchParams(location.search).get('selftest'))

    if (steps > 0) {
        await selfTest(game, redream, steps)
        return
    }

    window.addEventListener('keydown', event => {
        if (event.code in KEYS) {
            held.add(event.code)
            event.preventDefault()
        }
    })
    window.addEventListener('keyup', event => held.delete(event.code))
    window.addEventListener('blur', () => held.clear())

    const image = screen.createImageData(FRAME_W, FRAME_H)

    await redream.seed(game.render())
    show(game.render(), game.sprites.palette, image)
    status.textContent = 'arrows move, X jumps, Z runs'

    while (true) {
        const start = performance.now()

        // The keyboard is read once per step and held for the whole step.
        show(await redream.step(readKeys()), game.sprites.palette, image)

        // Capped at the game's rate; a slow machine just runs the game in slow motion.
        await new Promise(resolve => setTimeout(resolve, Math.max(STEP_MS - (performance.now() - start), 0)))
    }
}

main().catch(error => {
    status.textContent = `could not start: ${error.message}`
    console.error(error)
})
