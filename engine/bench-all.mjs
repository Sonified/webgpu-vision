#!/usr/bin/env node
/** Run engine/bench-all.html headless. Usage: node engine/bench-all.mjs */
import puppeteer from 'puppeteer';
import { createServer } from 'http';
import { readFileSync, existsSync } from 'fs';
import { join, extname } from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import { homedir } from 'os';
import { printBanner } from './session-timer.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');
const MIME = {'.html':'text/html','.js':'application/javascript','.mjs':'application/javascript','.wgsl':'text/plain','.json':'application/json','.bin':'application/octet-stream','.wasm':'application/wasm','.onnx':'application/octet-stream'};

const server = createServer((req, res) => {
  const urlPath = decodeURIComponent(req.url.split('?')[0]);
  if (urlPath.startsWith('/.session-timer/')) {
    const f = join(homedir(), urlPath);
    res.setHeader('Content-Type', 'text/plain');
    if (existsSync(f)) { res.writeHead(200); res.end(readFileSync(f, 'utf8')); }
    else { res.writeHead(404); res.end(''); }
    return;
  }
  let path = join(ROOT, urlPath);
  if (path.endsWith('/')) path += 'index.html';
  if (!existsSync(path)) { res.writeHead(404); res.end('Not found'); return; }
  res.setHeader('Cross-Origin-Embedder-Policy', 'credentialless');
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.writeHead(200, { 'Content-Type': MIME[extname(path)] || 'application/octet-stream' });
  res.end(readFileSync(path));
});

const PORT = 9445;

async function run() {
  await printBanner();
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
  page.on('console', m => {
    const text = m.text();
    if (text.includes('ERROR')) console.log('\x1b[31m' + text + '\x1b[0m');
    else if (text.includes('done')) console.log('\x1b[32m' + text + '\x1b[0m');
    else console.log(text);
  });
  page.on('pageerror', err => console.log('\x1b[31mPAGE ERROR: ' + err.message + '\x1b[0m'));

  console.log(`\nLoading bench-all.html (this takes a few minutes)...\n`);
  await page.goto(`http://localhost:${PORT}/engine/bench-all.html`, { waitUntil: 'networkidle0', timeout: 120000 });
  await page.waitForFunction(
    () => document.getElementById('log')?.textContent?.includes('done'),
    { timeout: 600000 }
  );

  await new Promise(r => setTimeout(r, 1000));
  await printBanner();
  await browser.close();
  server.close();
}

run().catch(err => { console.error(err); server.close(); process.exit(1); });
