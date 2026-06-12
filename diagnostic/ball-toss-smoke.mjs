// Smoke test: load ball-toss demo headless with fake camera, click Start,
// verify the render loop reaches RENDER ALIVE and keeps advancing frames.
import puppeteer from 'puppeteer';
import http from 'http';
import { promises as fs } from 'fs';
import path from 'path';

const ROOT = new URL('..', import.meta.url).pathname;
const PORT = 9444;

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
  } catch {
    res.writeHead(404); res.end('nf');
  }
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

const hits = { renderAlive: false, frame300: false, watchdog: false, errors: [] };
page.on('console', msg => {
  const t = msg.text();
  if (t.includes('RENDER ALIVE')) hits.renderAlive = true;
  if (t.includes('frame 300')) hits.frame300 = true;
  if (t.includes('[watchdog]')) { hits.watchdog = true; console.log('CONSOLE:', t); }
  if (msg.type() === 'error') { hits.errors.push(t); console.log('ERROR:', t); }
  if (t.includes('[lifecycle]') || t.includes('frozen')) console.log('CONSOLE:', t);
});
page.on('pageerror', e => { hits.errors.push(String(e)); console.log('PAGEERROR:', e); });

await page.goto(`http://localhost:${PORT}/demos/ball-toss/`, { waitUntil: 'domcontentloaded' });
await page.waitForSelector('#start-btn', { timeout: 10000 });
await new Promise(r => setTimeout(r, 1500));
await page.click('#start-btn');
await new Promise(r => setTimeout(r, 8000));

console.log('\n=== RESULT ===');
console.log('RENDER ALIVE:', hits.renderAlive);
console.log('frame 300 reached:', hits.frame300);
console.log('watchdog fired:', hits.watchdog);
console.log('page errors:', hits.errors.length ? hits.errors : 'none');

await browser.close();
server.close();
process.exit(hits.renderAlive && hits.errors.length === 0 ? 0 : 1);
