import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs'

// latent channels, context frames, latent height and width, buttons
const C = 8
const L = 8
const H = 28
const W = 32
const ACTION_DIM = 6

const WARMUP = 10
const FRAMES = 60
const BUDGET_MS = 50 // 20 frames per second

const params = new URLSearchParams(location.search)
const variants = (params.get('variants') || 'wide,mid,narrow').split(',')
const steps = Number(params.get('steps') || 2)

function log(line) {
    console.log(line)
    document.getElementById('log').textContent += line + '\n'
}

function randomTensor(dims) {
    const size = dims.reduce((a, b) => a * b, 1)
    const data = new Float32Array(size)

    for (let i = 0; i < size; i++) {
        data[i] = Math.random() * 2 - 1
    }

    return new ort.Tensor('float32', data, dims)
}

function median(times) {
    const sorted = [...times].sort((a, b) => a - b)
    return sorted[Math.floor(sorted.length / 2)]
}

async function bench(variant, half, decoder) {
    const pathCond = `cond_${variant}.onnx`
    const pathDenoiser = half ? `denoiser_${variant}_fp16.onnx` : `denoiser_${variant}.onnx`

    // The conditioning network is a few small matrix multiplies, so it runs on the CPU
    // and skips a round trip to the GPU.
    const cond = await ort.InferenceSession.create(pathCond, {executionProviders: ['wasm']});
    const denoiser = await ort.InferenceSession.create(pathDenoiser, {executionProviders: ['webgpu']});

    const context = randomTensor([1, L, C, H, W])
    const feedsCond = {
        tau: new ort.Tensor('float32', new Float32Array([1.0]), [1]),
        tau_ctx: new ort.Tensor('float32', new Float32Array([0.1]), [1]),
        actions: randomTensor([1, L + 1, ACTION_DIM]),
        level: new ort.Tensor('int64', new BigInt64Array([0n]), [1]),
        x_pos: new ort.Tensor('float32', new Float32Array([0.5]), [1]),
        powerup: new ort.Tensor('int64', new BigInt64Array([0n]), [1]),
        state_mask: new ort.Tensor('float32', new Float32Array([1.0]), [1])
    }

    const timesFrame = []
    const timesDenoise = []
    const timesDecode = []

    for (let frame = 0; frame < WARMUP + FRAMES; frame++) {
        const start = performance.now()

        // Each frame starts from noise and takes `steps` passes, the output of one pass
        // going back in as the input of the next. The values are meaningless with random
        // weights; the point is that the work matches a real frame.
        let z = randomTensor([1, C, H, W])

        for (let step = 0; step < steps; step++) {
            feedsCond.tau = new ort.Tensor('float32', new Float32Array([1.0 - step / steps]), [1])

            const resCond = await cond.run(feedsCond);
            const resDenoiser = await denoiser.run({z_noisy: z, context: context, cond: resCond.output});

            z = resDenoiser.output
        }

        const denoised = performance.now()

        await decoder.run({z: z});

        const end = performance.now()

        if (frame >= WARMUP) {
            timesFrame.push(end - start)
            timesDenoise.push(denoised - start)
            timesDecode.push(end - denoised)
        }
    }

    await cond.release();
    await denoiser.release();

    return {
        variant: variant,
        precision: half ? 'fp16' : 'fp32',
        frameMs: median(timesFrame),
        denoiseMs: median(timesDenoise),
        decodeMs: median(timesDecode),
        worstMs: Math.max(...timesFrame)
    }
}

async function main() {
    if (!navigator.gpu) {
        log('WebGPU is not available in this browser.')
        window.benchDone = true
        return
    }

    const adapter = await navigator.gpu.requestAdapter()
    log(`gpu: ${adapter.info.vendor} ${adapter.info.architecture} ${adapter.info.description}`)
    log(`${steps} denoise steps + 1 decode per frame, median of ${FRAMES} frames, budget ${BUDGET_MS} ms`)
    log('')

    const decoder = await ort.InferenceSession.create('decoder.onnx', {executionProviders: ['webgpu']});
    const results = []

    for (const variant of variants) {
        for (const half of [false, true]) {
            const result = await bench(variant, half, decoder)
            results.push(result)

            log(
                `${result.variant.padEnd(7)} ${result.precision}  ` +
                `frame ${result.frameMs.toFixed(1)} ms  ` +
                `(denoise ${result.denoiseMs.toFixed(1)}, decode ${result.decodeMs.toFixed(1)}, worst ${result.worstMs.toFixed(1)})  ` +
                `${(1000 / result.frameMs).toFixed(0)} fps  ` +
                `${result.frameMs <= BUDGET_MS ? 'fits' : 'too slow'}`
            )
        }
    }

    window.benchResults = results
    window.benchDone = true
}

main()
