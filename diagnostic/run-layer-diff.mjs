#!/usr/bin/env node
/**
 * Headless layer-by-layer diff: WGSL engine vs Python ORT reference.
 *
 * Prerequisites: run `python3 diagnostic/ort_dump.py images/test_images/hand_images/hand_000.png` first.
 *
 * Usage: node diagnostic/run-layer-diff.mjs [image_stem]
 *   e.g. node diagnostic/run-layer-diff.mjs hand_000
 */

import puppeteer from 'puppeteer';
import { createServer } from 'http';
import { readFileSync, existsSync } from 'fs';
import { join, extname } from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');
const imageStem = process.argv[2] || 'hand_000';
const exact = process.argv.includes('--exact');

const MIME = {
  '.html': 'text/html', '.js': 'application/javascript', '.mjs': 'application/javascript',
  '.wgsl': 'text/plain', '.json': 'application/json', '.bin': 'application/octet-stream',
  '.wasm': 'application/wasm', '.onnx': 'application/octet-stream', '.png': 'image/png',
};

const server = createServer((req, res) => {
  let urlPath = decodeURIComponent(req.url.split('?')[0]);
  let path = join(ROOT, urlPath);
  if (path.endsWith('/')) path += 'index.html';
  if (!existsSync(path)) { res.writeHead(404); res.end('Not found: ' + urlPath); return; }
  const ext = extname(path);
  res.setHeader('Cross-Origin-Embedder-Policy', 'credentialless');
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Content-Type', MIME[ext] || 'application/octet-stream');
  res.writeHead(200);
  res.end(readFileSync(path));
});

const PORT = 9333;

async function run() {
  // Verify ORT dump exists
  const dumpDir = join(ROOT, 'diagnostic', 'dumps', imageStem);
  if (!existsSync(join(dumpDir, '_manifest.json'))) {
    console.error(`No ORT dump found at ${dumpDir}/_manifest.json`);
    console.error(`Run: python3 diagnostic/ort_dump.py images/test_images/hand_images/${imageStem}.png`);
    process.exit(1);
  }

  await new Promise(r => server.listen(PORT, r));
  console.log(`Server on http://localhost:${PORT}`);

  const browser = await puppeteer.launch({
    headless: 'new',
    args: [
      '--enable-unsafe-webgpu',
      '--enable-features=Vulkan,UseSkiaRenderer',
      '--disable-gpu-sandbox',
      '--no-sandbox',
    ],
  });

  const page = await browser.newPage();

  page.on('console', msg => {
    const text = msg.text();
    if (text.includes('DIVERGE')) console.log('\x1b[31m' + text + '\x1b[0m');
    else if (text.includes('OK') && text.includes('[')) console.log('\x1b[32m' + text + '\x1b[0m');
    else if (text.includes('FIRST DIVERGENCE')) console.log('\x1b[1;31m' + text + '\x1b[0m');
    else if (text.includes('ALL LAYERS MATCH')) console.log('\x1b[1;32m' + text + '\x1b[0m');
    else console.log(text);
  });

  page.on('pageerror', err => console.log('\x1b[31mPAGE ERROR: ' + err.message + '\x1b[0m'));

  const url = `http://localhost:${PORT}/diagnostic/wgsl-dump.html?image=${imageStem}${exact ? '&exact=1' : ''}`;
  console.log(`\nLoading ${url}\n`);
  await page.goto(url, { waitUntil: 'networkidle0', timeout: 60000 });

  await page.waitForFunction(
    () => window.__diffDone === true || window.__diffDone === 'error',
    { timeout: 120000 }
  );

  await new Promise(r => setTimeout(r, 500));

  const status = await page.evaluate(() => window.__diffDone);
  await browser.close();
  server.close();
  process.exit(status === true ? 0 : 1);
}

run().catch(err => { console.error(err); process.exit(1); });
