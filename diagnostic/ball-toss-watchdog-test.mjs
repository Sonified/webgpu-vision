// Watchdog test: kill requestAnimationFrame after 5 deliveries to simulate
// Chrome's render suppression (COI SW reload, no user activation). The
// watchdog must detect the stall, log, and pump frames so the render loop
// reaches RENDER ALIVE (frame 10) without rAF.
import puppeteer from 'puppeteer';
import http from 'http';
import { promises as fs } from 'fs';
import path from 'path';

const ROOT = new URL('..', import.meta.url).pathname;
const PORT = 9445;
const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.json': 'application/json', '.bin': 'application/octet-stream',
  '.wasm': 'application/wasm', '.onnx': 'application/octet-stream',
  '.png': 'image/png', '.css': 'text/css',
};
const server = http.createServer(async (req, res) => {
  try {
    const urlPath = decodeURIComponent(new URL(req.url, 'http://x').pathname);
    let file = path.join(ROOT, urlPath);
    if ((await fs.stat(file).catch(() => null))?.isDirectory()) file = path.join(file, 'index.html');
    const data = await fs.readFile(file);
    res.writeHead(200, {
      'Content-Type': MIME[path.extname(file)] || 'application/octet-stream',
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Resource-Policy': 'cross-origin',
    });
    res.end(data);
  } catch { res.writeHead(404); res.end('nf'); }
});
await new Promise(r => server.listen(PORT, r));

const browser = await puppeteer.launch({
  headless: 'new',
  args: [
    '--enable-unsafe-webgpu', '--enable-features=Vulkan',
    '--no-sandbox', '--disable-gpu-sandbox',
    '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
  ],
});
const page = await browser.newPage();

// Sabotage rAF: deliver 5 callbacks normally, then accept registrations
// but never fire them (Chrome's full-suppression behavior).
await page.evaluateOnNewDocument(() => {
  const realRaf = window.requestAnimationFrame.bind(window);
  let delivered = 0;
  window.requestAnimationFrame = (cb) => {
    if (delivered >= 5) return 999999; // swallow: registered, never fires
    return realRaf((ts) => { delivered++; cb(ts); });
  };
});

const hits = { renderAlive: false, watchdogStall: false, pumpOff: false, errors: [] };
page.on('console', msg => {
  const t = msg.text();
  if (t.includes('RENDER ALIVE')) hits.renderAlive = true;
  if (t.includes('render loop stalled')) hits.watchdogStall = true;
  if (t.includes('pump off')) hits.pumpOff = true;
  if (msg.type() === 'error' && !t.includes('onnxruntime')) hits.errors.push(t);
  if (t.includes('[watchdog]') || t.includes('animate()') || t.includes('frozen')) console.log('CONSOLE:', t);
});
page.on('pageerror', e => { hits.errors.push(String(e)); console.log('PAGEERROR:', e); });

await page.goto(`http://localhost:${PORT}/demos/ball-toss/`, { waitUntil: 'domcontentloaded' });
await page.waitForSelector('#start-btn', { timeout: 10000 });
await new Promise(r => setTimeout(r, 1000));
await page.click('#start-btn');
await new Promise(r => setTimeout(r, 7000));

console.log('\n=== RESULT ===');
console.log('watchdog detected stall:', hits.watchdogStall);
console.log('RENDER ALIVE via pump:', hits.renderAlive);
console.log('errors:', hits.errors.length ? hits.errors : 'none');
await browser.close();
server.close();
process.exit(hits.watchdogStall && hits.renderAlive && hits.errors.length === 0 ? 0 : 1);
