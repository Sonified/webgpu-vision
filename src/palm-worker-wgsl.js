// Palm detection worker: WGSL engine (replaces ORT).
// Receives ImageBitmap, does GPU letterbox + WGSL inference + decode + NMS.

import { applyLogGatesFromUrl, log } from './log-gates.js';
import { ModelRunner } from '../engine/model-runner.js';
import { generateAnchors, decodeDetections } from './anchors.js';
import { weightedNMS } from './nms.js';

// Load gate state BEFORE any other code in this worker runs so the first
// gated log (e.g. worker ready) respects the user's preferences. ES imports
// are evaluated before this line but none log at evaluation time.
applyLogGatesFromUrl();

const PALM_SIZE = 192;
const MODEL_JSON_URL = '../models/palm_detection_lite.json';
const MODEL_BIN_URL = '../models/palm_detection_lite.bin';

let runner = null;
let anchors = null;
let device = null;
let inputBuf = null;
let compiled = false;
let previewReadBuf = null;

// GPU letterbox (same as before -- preprocesses camera frame to model input)
let letterboxPipeline = null;
let letterboxOutputBuf = null;
let uniformBuffer = null;
let sampler = null;

const WGSL_LETTERBOX = `
struct Uniforms {
  scale: f32,
  offset_x: f32,
  offset_y: f32,
  src_w: f32,
  src_h: f32,
}

@group(0) @binding(0) var srcTexture: texture_2d<f32>;
@group(0) @binding(1) var srcSampler: sampler;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> u: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let x = gid.x;
  let y = gid.y;
  if (x >= ${PALM_SIZE}u || y >= ${PALM_SIZE}u) { return; }

  let fx = f32(x) + 0.5;
  let fy = f32(y) + 0.5;
  let sx = (fx - u.offset_x) / u.scale;
  let sy = (fy - u.offset_y) / u.scale;

  var r = 0.0; var g = 0.0; var b = 0.0;
  if (sx >= 0.0 && sx < u.src_w && sy >= 0.0 && sy < u.src_h) {
    let uv = vec2f(sx / u.src_w, sy / u.src_h);
    let pixel = textureSampleLevel(srcTexture, srcSampler, uv, 0.0);
    r = pixel.r;
    g = pixel.g;
    b = pixel.b;
  }

  // Output as NCHW: channel-first layout (engine compiles Transpose out)
  let spatial = ${PALM_SIZE}u * ${PALM_SIZE}u;
  let idx = y * ${PALM_SIZE}u + x;
  output[idx]               = r;
  output[spatial + idx]     = g;
  output[2u * spatial + idx] = b;
}
`;

async function initGPU() {
  const adapter = await navigator.gpu.requestAdapter();
  device = await adapter.requestDevice();

  // Letterbox pipeline
  letterboxPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: device.createShaderModule({ code: WGSL_LETTERBOX }), entryPoint: 'main' },
  });

  const BF = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  letterboxOutputBuf = device.createBuffer({ size: PALM_SIZE * PALM_SIZE * 3 * 4, usage: BF });
  inputBuf = letterboxOutputBuf; // letterbox output IS the model input

  uniformBuffer = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  previewReadBuf = device.createBuffer({ size: PALM_SIZE * PALM_SIZE * 3 * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

  // Load model
  const graph = await (await fetch(MODEL_JSON_URL)).json();
  const allWeights = new Float32Array(await (await fetch(MODEL_BIN_URL)).arrayBuffer());

  // Create weight buffers
  const W = {};
  for (const [name, info] of Object.entries(graph.weights)) {
    if (info.length === 0) continue;
    const buf = device.createBuffer({ size: Math.max(info.length * 4, 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buf, 0, allWeights.subarray(info.offset, info.offset + info.length));
    W[name] = buf;
  }

  // Consolidated weight buffer for fused shader
  const allWeightsBuf = device.createBuffer({ size: allWeights.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(allWeightsBuf, 0, allWeights);

  // Create pipelines
  const mkP = async (name) => {
    const code = await (await fetch(`../engine/${name}.wgsl`)).text();
    return device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } });
  };
  const P = {
    conv2d: await mkP('conv2d'), maxpool: await mkP('maxpool'), resize: await mkP('resize'),
    gemm: await mkP('gemm'), global_avg_pool: await mkP('global_avg_pool'),
    add: await mkP('add'), pad_channels: await mkP('pad_channels'),
    fused_block: await mkP('fused_block'), transpose_nhwc: await mkP('transpose_nhwc'),
  };

  // Create runner and compile
  runner = new ModelRunner(device, P, W, allWeightsBuf);
  await runner.compile(graph, inputBuf, allWeights);
  compiled = true;

  return graph;
}

// Cached letterbox GPU resources -- reused every frame (only recreated if video resolution changes)
let cachedLBTexture = null;
let cachedLBBindGroup = null;
let cachedLBSize = [0, 0];
let cachedLetterbox = null;

function gpuLetterbox(source) {
  const srcW = source.displayWidth || source.width, srcH = source.displayHeight || source.height;
  const scale = PALM_SIZE / Math.max(srcW, srcH);
  const dstW = Math.round(srcW * scale), dstH = Math.round(srcH * scale);
  const offsetX = (PALM_SIZE - dstW) / 2, offsetY = (PALM_SIZE - dstH) / 2;

  // Recreate texture + bind group only when dimensions change
  if (srcW !== cachedLBSize[0] || srcH !== cachedLBSize[1]) {
    if (cachedLBTexture) cachedLBTexture.destroy();
    cachedLBTexture = device.createTexture({
      size: [srcW, srcH], format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    cachedLBBindGroup = device.createBindGroup({
      layout: letterboxPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: cachedLBTexture.createView() },
        { binding: 1, resource: sampler },
        { binding: 2, resource: { buffer: letterboxOutputBuf } },
        { binding: 3, resource: { buffer: uniformBuffer } },
      ],
    });
    cachedLBSize = [srcW, srcH];
    cachedLetterbox = {
      scaleX: dstW / PALM_SIZE, scaleY: dstH / PALM_SIZE,
      offsetX: offsetX / PALM_SIZE, offsetY: offsetY / PALM_SIZE,
    };
    // Uniforms only change when dimensions change
    device.queue.writeBuffer(uniformBuffer, 0, new Float32Array([scale, offsetX, offsetY, srcW, srcH, 0, 0, 0]));
  }

  device.queue.copyExternalImageToTexture({ source }, { texture: cachedLBTexture }, [srcW, srcH]);

  return cachedLetterbox;
}

function encodeLetterbox(enc) {
  const pass = enc.beginComputePass();
  pass.setPipeline(letterboxPipeline);
  pass.setBindGroup(0, cachedLBBindGroup);
  pass.dispatchWorkgroups(Math.ceil(PALM_SIZE / 16), Math.ceil(PALM_SIZE / 16));
  pass.end();
}

self.onmessage = async (e) => {
  const { type } = e.data;

  if (type === 'init') {
    try {
      anchors = generateAnchors();
      await initGPU();
      log('lifecycle', '[palm-worker-wgsl] ready (compiled WGSL engine)');
      self.postMessage({ type: 'ready', gpuLetterbox: true, gpuDirect: true });
    } catch (err) {
      console.error('[palm-worker-wgsl] init error:', err);
      self.postMessage({ type: 'error', message: err.message });
    }
  }

  if (type === 'detect') {
    try {
      const { frame } = e.data;

      // GPU letterbox: upload frame + write uniforms
      const letterbox = gpuLetterbox(frame);
      frame.close();

      // Single encoder: letterbox + inference + readback in one submit
      const enc = device.createCommandEncoder();
      encodeLetterbox(enc);

      runner.encodeInto(enc);
      enc.copyBufferToBuffer(letterboxOutputBuf, 0, previewReadBuf, 0, previewReadBuf.size);
      device.queue.submit([enc.finish()]);
      const outputs = await runner.readOutputs();

      // Read back letterbox for preview (NCHW float32 -> RGBA uint8)
      let previewRGBA = null;
      try {
        await previewReadBuf.mapAsync(GPUMapMode.READ);
        const nchw = new Float32Array(previewReadBuf.getMappedRange().slice(0));
        previewReadBuf.unmap();
        const S = PALM_SIZE * PALM_SIZE;
        const rgba = new Uint8ClampedArray(S * 4);
        for (let i = 0; i < S; i++) {
          rgba[i * 4]     = nchw[i] * 255;
          rgba[i * 4 + 1] = nchw[S + i] * 255;
          rgba[i * 4 + 2] = nchw[2 * S + i] * 255;
          rgba[i * 4 + 3] = 255;
        }
        previewRGBA = rgba.buffer;
      } catch (_) {}

      // Decode outputs -- find regressors (dim 18) and scores (dim 1)
      let regressors, scores;
      for (const [name, data] of Object.entries(outputs)) {
        if (data.length > 2016) regressors = data; // 2016 * 18 or 2016 * 1
        else scores = data;
      }

      // If outputs are in a concatenated format, split them
      // Palm model outputs: regressors [1, 2016, 18] and classificators [1, 2016, 1]
      if (!scores && regressors) {
        // Might be named differently -- check by size
        for (const [name, data] of Object.entries(outputs)) {
          if (data.length === 2016) scores = data;
          else if (data.length === 2016 * 18) regressors = data;
        }
      }

      let detections = decodeDetections(regressors, scores, anchors);
      const preNMS = detections.length;
      const preDetail = detections.map(d => `(${(d.cx*100).toFixed(0)},${(d.cy*100).toFixed(0)} ${(d.w*100).toFixed(0)}x${(d.h*100).toFixed(0)} s=${d.score.toFixed(2)})`).join(' ');
      detections = weightedNMS(detections);
      if (preNMS > 0) {
        console.log(`[palm-det] pre(${preNMS}): ${preDetail} -> post(${detections.length}): ${detections.map(d => `(${(d.cx*100).toFixed(0)},${(d.cy*100).toFixed(0)})`).join(' ')}`);
      }

      const msg = { type: 'detections', detections, letterbox };
      const transfer = [];
      if (previewRGBA) { msg.previewRGBA = previewRGBA; transfer.push(previewRGBA); }
      self.postMessage(msg, transfer);
    } catch (err) {
      self.postMessage({ type: 'error', message: err.message });
    }
  }
};
