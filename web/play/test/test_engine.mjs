// Holds engine.js to simul/engine.py: for the buttons recorded in trace.json (written by
// export/export_web.py) every frame must come out identical. Run: node web/play/test/test_engine.mjs
import {Game} from '../engine.js'
import {readFileSync} from 'node:fs'
import {gunzipSync, crc32} from 'node:zlib'

const here = new URL('.', import.meta.url).pathname
const index = JSON.parse(readFileSync(here + '../assets.json'))
const blob = new Uint8Array(gunzipSync(readFileSync(here + '../assets.bin')))
const traces = JSON.parse(readFileSync(here + 'trace.json'))

for (const trace of traces) {
    const game = new Game(index, blob, trace.level)

    trace.actions.forEach((action, step) => {
        const frame = game.step(action)

        if (crc32(frame) !== trace.checksums[step]) {
            console.log(`level ${trace.level} step ${step}: frame differs from the Python engine (x ${game.x} y ${game.y} mode ${game.mode})`)
            process.exit(1)
        }
    })

    console.log(`level ${trace.level}: ${trace.actions.length} frames identical to the Python engine`)
}

console.log('\nAll tests passed.')
