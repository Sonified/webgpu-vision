// Face landmark worker: WGSL engine (replaces ORT).
// Receives ImageBitmap + rect, does GPU affine warp + WGSL inference -> 478 landmarks.

import { applyLogGatesFromUrl, log } from './log-gates.js';
import { ModelRunner } from '../engine/model-runner.js';

applyLogGatesFromUrl();

const S = 256; // face landmark model input size
const NUM_LANDMARKS = 478;
const LANDMARK_FLOATS = NUM_LANDMARKS * 3; // 1434

const MODEL_BASE = '../models/face/face_landmarks/face_landmarks_detector';

let runner = null;
let device = null;
let inputBuf = null;
let hasF16 = false;
let useF16 = false;

let warpPipeline = null;
let uniformBuffer = null;
let gpuSampler = null;
let outputNames = {};

// f32->f16 cast state
let castPipeline = null;
let castBindGroup = null;
let castParamsBuf = null;
let inputBufF16 = null;

// Pipeline caches
let pipelinesF32 = null;
let pipelinesF16 = null;

const WGSL_WARP = `
struct Uniforms {
  a: f32, b: f32, c: f32, _pad0: f32,
  d: f32, e: f32, f_: f32, _pad1: f32,
  src_w: f32, src_h: f32,
}

@group(0) @binding(0) var srcTexture: texture_2d<f32>;
@group(0) @binding(1) var srcSampler: sampler;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> u: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let x = gid.x;
  let y = gid.y;
  if (x >= ${S}u || y >= ${S}u) { return; }

  let fx = f32(x) + 0.5;
  let fy = f32(y) + 0.5;
  let sx = u.a * fx + u.b * fy + u.c;
  let sy = u.d * fx + u.e * fy + u.f_;
  let uv = vec2f(sx / u.src_w, sy / u.src_h);
  let pixel = textureSampleLevel(srcTexture, srcSampler, uv, 0.0);

  // Output NCHW
  let spatial = ${S}u * ${S}u;
  let idx = y * ${S}u + x;
  output[idx]               = pixel.r;
  output[spatial + idx]     = pixel.g;
  output[2u * spatial + idx] = pixel.b;
}
`;

function rotatedRectPoints(cx, cy, w, h, rotation) {
  const b = Math.cos(rotation) * 0.5;
  const a = Math.sin(rotation) * 0.5;
  const p0x = cx - a * h - b * w, p0y = cy + b * h - a * w;
  const p1x = cx + a * h - b * w, p1y = cy - b * h - a * w;
  return [[p0x, p0y], [p1x, p1y], [2 * cx - p0x, 2 * cy - p0y], [2 * cx - p1x, 2 * cy - p1y]];
}

function computeAffineParams(rect) {
  const pts = rotatedRectPoints(rect.cx, rect.cy, rect.w, rect.h, rect.angle);
  const [p1x, p1y] = pts[1];
  const [p2x, p2y] = pts[2];
  const [p3x, p3y] = pts[3];
  const dx1 = p2x - p1x, dy1 = p2y - p1y;
  const dx2 = p3x - p1x, dy2 = p3y - p1y;
  const det = dx1 * dy2 - dx2 * dy1;
  if (Math.abs(det) < 1e-6) return null;
  const invDet = 1 / det;
  const fa = S * (dy2 - dy1) * invDet;
  const fb = S * (dx1 - dx2) * invDet;
  const fd = S * (-dy1) * invDet;
  const fe = S * dx1 * invDet;
  const fc = -fa * p1x - fb * p1y;
  const ff = -fd * p1x - fe * p1y;
  const detM = fa * fe - fb * fd;
  return {
    a: fe / detM, b: -fb / detM,
    c: -(fe * fc - fb * ff) / detM,
    d: -fd / detM, e: fa / detM,
    f: -(-fd * fc + fa * ff) / detM,
  };
}

async function initGPU(wantF16) {
  const adapter = await navigator.gpu.requestAdapter();
  hasF16 = adapter.features.has('shader-f16');
  const features = hasF16 ? ['shader-f16'] : [];
  device = await adapter.requestDevice({ requiredFeatures: features });

  warpPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: device.createShaderModule({ code: WGSL_WARP }), entryPoint: 'main' },
  });

  const BF = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  inputBuf = device.createBuffer({ size: S * S * 3 * 4, usage: BF });
  uniformBuffer = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  gpuSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

  const mkP = async (name) => {
    const code = await (await fetch(`../engine/${name}.wgsl`)).text();
    return device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } });
  };
  pipelinesF32 = {
    conv2d: await mkP('conv2d'), maxpool: await mkP('maxpool'), resize: await mkP('resize'),
    gemm: await mkP('gemm'), global_avg_pool: await mkP('global_avg_pool'),
    add: await mkP('add'), pad_channels: await mkP('pad_channels'),
    fused_block: await mkP('fused_block'), fused_block_tiled: await mkP('fused_block_tiled'),
    transpose_nhwc: await mkP('transpose_nhwc'),
  };

  if (hasF16) {
    const mkF16 = async (name) => {
      const code = await (await fetch(`../engine/${name}_f16.wgsl`)).text();
      return device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } });
    };
    pipelinesF16 = {
      conv2d: await mkF16('conv2d'), maxpool: await mkF16('maxpool'),
      gemm: await mkF16('gemm'), global_avg_pool: await mkF16('global_avg_pool'),
      add: await mkF16('add'), fused_block: await mkF16('fused_block'),
      fused_block_tiled: null,
      resize: pipelinesF32.resize,
      transpose_nhwc: await mkF16('transpose_nhwc'),
      pad_channels: await mkF16('pad_channels'),
    };
    const castCode = await (await fetch('../engine/cast_f32_to_f16.wgsl')).text();
    castPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: castCode }), entryPoint: 'main' },
    });
    castParamsBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(castParamsBuf, 0, new Uint32Array([S * S * 3]));
  }

  await loadModel(wantF16 && hasF16);
}

async function loadModel(f16) {
  useF16 = f16;
  const suffix = f16 ? '_f16' : '';
  const bpe = f16 ? 2 : 4;
  const TA = f16 ? Float16Array : Float32Array;
  const P = f16 ? pipelinesF16 : pipelinesF32;

  const graph = await (await fetch(`${MODEL_BASE}${suffix}.json`)).json();
  const allWeights = new TA(await (await fetch(`${MODEL_BASE}${suffix}.bin`)).arrayBuffer());

  const align4 = (arr) => {
    if (arr.byteLength % 4 === 0) return arr;
    const padded = new TA(arr.length + 1);
    padded.set(arr);
    return padded;
  };

  const W = {};
  for (const [name, info] of Object.entries(graph.weights)) {
    if (info.length === 0) continue;
    const data = align4(allWeights.subarray(info.offset, info.offset + info.length));
    const buf = device.createBuffer({ size: Math.max(data.byteLength, 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buf, 0, data);
    W[name] = buf;
  }

  const allWPadded = align4(allWeights);
  const allWeightsBuf = device.createBuffer({ size: allWPadded.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(allWeightsBuf, 0, allWPadded);

  let modelInputBuf = inputBuf;
  if (f16) {
    if (!inputBufF16) {
      inputBufF16 = device.createBuffer({
        size: S * S * 3 * 2 + (S * S * 3 * 2 % 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
    }
    modelInputBuf = inputBufF16;
    castBindGroup = device.createBindGroup({
      layout: castPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: castParamsBuf } },
        { binding: 1, resource: { buffer: inputBuf } },
        { binding: 2, resource: { buffer: inputBufF16 } },
      ],
    });
  } else {
    castBindGroup = null;
  }

  const newRunner = new ModelRunner(device, P, W, allWeightsBuf, f16 ? { f16: true } : {});
  const outputs = await newRunner.compile(graph, modelInputBuf, allWeights);
  runner = newRunner;

  outputNames = {};
  for (const [name, data] of Object.entries(outputs)) {
    if (data.length === LANDMARK_FLOATS) outputNames.landmarks = name;
    else if (data.length === 1 && !outputNames.faceFlag) outputNames.faceFlag = name;
  }
}

// Cached warp GPU resources -- reused every frame (only recreated if video resolution changes)
let cachedWarpTexture = null;
let cachedWarpBindGroup = null;
let cachedWarpSize = [0, 0];
const warpUniforms = new Float32Array(12);

function dispatchWarp(source, inv) {
  const w = source.displayWidth || source.width, h = source.displayHeight || source.height;

  if (w !== cachedWarpSize[0] || h !== cachedWarpSize[1]) {
    if (cachedWarpTexture) cachedWarpTexture.destroy();
    cachedWarpTexture = device.createTexture({
      size: [w, h], format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    cachedWarpBindGroup = device.createBindGroup({
      layout: warpPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: cachedWarpTexture.createView() },
        { binding: 1, resource: gpuSampler },
        { binding: 2, resource: { buffer: inputBuf } },
        { binding: 3, resource: { buffer: uniformBuffer } },
      ],
    });
    cachedWarpSize = [w, h];
  }

  device.queue.copyExternalImageToTexture({ source }, { texture: cachedWarpTexture }, [w, h]);
  warpUniforms[0] = inv.a; warpUniforms[1] = inv.b; warpUniforms[2] = inv.c;
  warpUniforms[4] = inv.d; warpUniforms[5] = inv.e; warpUniforms[6] = inv.f;
  warpUniforms[8] = w; warpUniforms[9] = h;
  device.queue.writeBuffer(uniformBuffer, 0, warpUniforms);
}

function encodeWarp(enc) {
  const pass = enc.beginComputePass();
  pass.setPipeline(warpPipeline);
  pass.setBindGroup(0, cachedWarpBindGroup);
  pass.dispatchWorkgroups(Math.ceil(S / 16), Math.ceil(S / 16));
  pass.end();
}

self.onmessage = async (e) => {
  const { type } = e.data;

  if (type === 'init') {
    try {
      await initGPU(e.data.f16);
      const tag = useF16 ? 'f16' : 'f32';
      log('lifecycle', `[face-lm-worker-wgsl] ready (WGSL ${tag})`);
      self.postMessage({ type: 'ready', gpuDirect: true, f16: useF16 });
    } catch (err) {
      console.error('[face-lm-worker-wgsl] init error:', err);
      self.postMessage({ type: 'error', message: err.message });
    }
  }

  if (type === 'switchPrecision') {
    try {
      const wantF16 = e.data.f16 && hasF16;
      if (wantF16 === useF16) { self.postMessage({ type: 'switchDone', f16: useF16 }); return; }
      await loadModel(wantF16);
      log('lifecycle', `[face-lm-worker-wgsl] switched to ${useF16 ? 'f16' : 'f32'}`);
      self.postMessage({ type: 'switchDone', f16: useF16 });
    } catch (err) {
      console.error('[face-lm-worker-wgsl] switchPrecision error:', err);
      self.postMessage({ type: 'error', message: err.message });
    }
  }

  if (type === 'infer') {
    try {
      const { frame, rect, vw, vh } = e.data;

      const inv = computeAffineParams(rect);
      if (!inv) {
        frame.close();
        self.postMessage({ type: 'result', faceFlag: 0, landmarks: null });
        return;
      }

      dispatchWarp(frame, inv);
      frame.close();

      const enc = device.createCommandEncoder();
      encodeWarp(enc);
      if (useF16 && castBindGroup) {
        const castPass = enc.beginComputePass();
        castPass.setPipeline(castPipeline);
        castPass.setBindGroup(0, castBindGroup);
        castPass.dispatchWorkgroups(Math.ceil(S * S * 3 / 256));
        castPass.end();
      }
      runner.encodeInto(enc);
      device.queue.submit([enc.finish()]);
      const outputs = await runner.readOutputs();

      const rawLandmarks = outputNames.landmarks ? outputs[outputNames.landmarks] : null;
      const faceFlag = outputNames.faceFlag ? outputs[outputNames.faceFlag][0] : 0;

      // Project 478 landmarks back to video space
      let projectedLandmarks = null;
      if (rawLandmarks && rawLandmarks.length === LANDMARK_FLOATS) {
        projectedLandmarks = new Float32Array(LANDMARK_FLOATS);
        for (let i = 0; i < NUM_LANDMARKS; i++) {
          const ox = rawLandmarks[i * 3];
          const oy = rawLandmarks[i * 3 + 1];
          const oz = rawLandmarks[i * 3 + 2];
          projectedLandmarks[i * 3]     = (inv.a * ox + inv.b * oy + inv.c) / vw;
          projectedLandmarks[i * 3 + 1] = (inv.d * ox + inv.e * oy + inv.f) / vh;
          projectedLandmarks[i * 3 + 2] = oz / S;
        }
      }

      const rawCopy = rawLandmarks ? new Float32Array(rawLandmarks).buffer : null;
      const transferables = [];
      if (projectedLandmarks) transferables.push(projectedLandmarks.buffer);
      if (rawCopy) transferables.push(rawCopy);

      self.postMessage({
        type: 'result',
        faceFlag,
        landmarks: projectedLandmarks ? projectedLandmarks.buffer : null,
        rawLandmarks: rawCopy,
        modelSize: S,
      }, transferables);
    } catch (err) {
      self.postMessage({ type: 'error', message: err.message });
    }
  }
};
