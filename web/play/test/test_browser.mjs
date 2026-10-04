// Runs the page's self-test in headless Chrome: the real models on WebGPU, a fixed run
// of buttons, and a report of the time per frame and how closely the model's frames
// follow the simulation's. Also saves a screenshot of the last frame.
// Run from the project root: node web/play/test/test_browser.mjs [steps]
import {spawn} from 'node:child_process'
import {mkdtempSync, writeFileSync} from 'node:fs'
import {tmpdir} from 'node:os'

const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
const PORT = 8123
const DEBUG_PORT = 9333
const TIMEOUT_MS = 180000

const steps = Number(process.argv[2] || 200)
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms))

const server = spawn('python3', ['-m', 'http.server', String(PORT), '--directory', 'web/play'], {stdio: 'ignore'})
const chrome = spawn(CHROME, [
    '--headless=new', '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=metal',
    `--remote-debugging-port=${DEBUG_PORT}`, `--user-data-dir=${mkdtempSync(tmpdir() + '/chrome-')}`,
    '--window-size=900,900', `http://localhost:${PORT}/index.html?selftest=${steps}`
], {stdio: 'ignore'})

function finish(code) {
    chrome.kill()
    server.kill()
    process.exit(code)
}

let target = null

for (let tries = 0; tries < 50 && target === null; tries++) {
    await sleep(200)

    try {
        const pages = await (await fetch(`http://localhost:${DEBUG_PORT}/json`)).json()
        target = pages.find(page => page.type === 'page') || null
    } catch (error) {
        // Chrome is not listening yet.
    }
}

if (target === null) {
    console.log('Chrome did not start')
    finish(1)
}

const socket = new WebSocket(target.webSocketDebuggerUrl)
const waiting = new Map()
let nextId = 1

socket.onmessage = event => {
    const message = JSON.parse(event.data)

    if (message.id && waiting.has(message.id)) {
        waiting.get(message.id)(message.result)
        waiting.delete(message.id)
    }
    if (message.method === 'Runtime.exceptionThrown') {
        console.log('page error:', message.params.exceptionDetails.exception?.description || message.params.exceptionDetails.text)
    }
    if (message.method === 'Runtime.consoleAPICalled' && message.params.type === 'error') {
        console.log('console error:', message.params.args.map(arg => arg.value || arg.description).join(' '))
    }
}

await new Promise(resolve => socket.onopen = resolve)

function send(method, params = {}) {
    return new Promise(resolve => {
        waiting.set(nextId, resolve)
        socket.send(JSON.stringify({id: nextId++, method: method, params: params}))
    })
}

await send('Runtime.enable')

const start = Date.now()
let text = ''

while (Date.now() - start < TIMEOUT_MS) {
    await sleep(1000)

    const result = await send('Runtime.evaluate', {expression: "document.title + '|' + document.getElementById('status').textContent"})
    text = result.result.value

    if (text.startsWith('selftest done') || text.includes('could not start') || text.includes('needs WebGPU')) {
        break
    }
}

console.log(text.split('|')[1])

const shot = await send('Page.captureScreenshot', {format: 'png'})
writeFileSync('data/samples/web_play.png', Buffer.from(shot.data, 'base64'))
console.log('wrote data/samples/web_play.png')

const passed = text.startsWith('selftest done')
console.log(passed ? '\nAll tests passed.' : '\nThe page did not finish its self-test.')
finish(passed ? 0 : 1)
