#!/usr/bin/env node
/** Run engine/profile-face-det.html headless. Usage: node engine/profile-face-det.mjs */
import puppeteer from 'puppeteer';
import { createServer } from 'http';
import { readFileSync, existsSync } from 'fs';
import { join, extname, dirname } from 'path';
import { fileURLToPath } from 'url';
import { homedir } from 'os';
import { printBanner } from './session-timer.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');
const MIME = {'.html':'text/html','.js':'application/javascript','.mjs':'application/javascript','.wgsl':'text/plain','.json':'application/json','.bin':'application/octet-stream','.wasm':'application/wasm'};

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

const PORT = 9446;

async function run() {
  await printBanner();
  await new Promise(r => server.listen(PORT, r));

  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,UseSkiaRenderer', '--disable-gpu-sandbox', '--no-sandbox'],
  });

  const page = await browser.newPage();
  page.on('console', m => console.log(m.text()));
  page.on('pageerror', err => console.log('PAGE ERROR: ' + err.message));

  await page.goto(`http://localhost:${PORT}/engine/profile-face-det.html`, { waitUntil: 'networkidle0', timeout: 60000 });
  await page.waitForFunction(() => document.getElementById('log')?.textContent?.includes('done'), { timeout: 120000 });

  await new Promise(r => setTimeout(r, 500));
  await printBanner();
  await browser.close();
  server.close();
}

run().catch(err => { console.error(err); server.close(); process.exit(1); });
