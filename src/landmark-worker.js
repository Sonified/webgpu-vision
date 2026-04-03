// Landmark inference worker: receives ImageBitmap + rect, does crop + preprocess + inference.
// Each worker owns its own WASM instance, WebGPU device, and warp canvas.

import * as ort from 'onnxruntime-web/webgpu';

ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;

const LANDMARK_SIZE = 224;

let session = null;
let outputMap = null;

// Preallocated buffers -- reused every frame, zero allocation
const warpCanvas = new OffscreenCanvas(LANDMARK_SIZE, LANDMARK_SIZE);
const warpCtx = warpCanvas.getContext('2d', { willReadFrequently: true });
const inputBuffer = new Float32Array(LANDMARK_SIZE * LANDMARK_SIZE * 3);

function rotatedRectPoints(cx, cy, w, h, rotation) {
  const b = Math.cos(rotation) * 0.5;
  const a = Math.sin(rotation) * 0.5;
  const p0x = cx - a * h - b * w, p0y = cy + b * h - a * w;
  const p1x = cx + a * h - b * w, p1y = cy - b * h - a * w;
  return [[p0x, p0y], [p1x, p1y], [2 * cx - p0x, 2 * cy - p0y], [2 * cx - p1x, 2 * cy - p1y]];
}

function cropAndPreprocess(bitmap, rect) {
  const S = LANDMARK_SIZE;
  const pts = rotatedRectPoints(rect.cx, rect.cy, rect.w, rect.h, rect.angle);
  const [p1x, p1y] = pts[1];
  const [p2x, p2y] = pts[2];
  const [p3x, p3y] = pts[3];

  const dx1 = p2x - p1x, dy1 = p2y - p1y;
  const dx2 = p3x - p1x, dy2 = p3y - p1y;
  const det = dx1 * dy2 - dx2 * dy1;
  if (Math.abs(det) < 1e-6) return null;

  const invDet = 1 / det;
  const a = S * (dy2 - dy1) * invDet;
  const b = S * (dx1 - dx2) * invDet;
  const d = S * (-dy1) * invDet;
  const e = S * dx1 * invDet;
  const c = -a * p1x - b * p1y;
  const f = -d * p1x - e * p1y;

  // Affine warp
  warpCtx.resetTransform();
  warpCtx.clearRect(0, 0, S, S);
  warpCtx.setTransform(a, d, b, e, c, f);
  warpCtx.drawImage(bitmap, 0, 0);
  warpCtx.resetTransform();

  // Read pixels and normalize to [0,1] into preallocated buffer
  const imageData = warpCtx.getImageData(0, 0, S, S);
  const rgba = imageData.data;
  const pixelCount = S * S;
  for (let i = 0; i < pixelCount; i++) {
    const ri = i * 4;
    const oi = i * 3;
    inputBuffer[oi]     = rgba[ri]     / 255;
    inputBuffer[oi + 1] = rgba[ri + 1] / 255;
    inputBuffer[oi + 2] = rgba[ri + 2] / 255;
  }

  // Inverse affine for projecting landmarks back
  const detM = a * e - b * d;
  const inverseTransform = {
    a: e / detM, b: -b / detM,
    c: -(e * c - b * f) / detM,
    d: -d / detM, e: a / detM,
    f: -(-d * c + a * f) / detM,
  };

  return inverseTransform;
}

self.onmessage = async (e) => {
  const { type } = e.data;

  if (type === 'init') {
    try {
      session = await ort.InferenceSession.create(e.data.modelUrl, {
        executionProviders: ['webgpu'],
        graphOptimizationLevel: 'all',
      });

      // Warmup + discover output names
      const warmup = new ort.Tensor('float32', new Float32Array(224 * 224 * 3), [1, 224, 224, 3]);
      const results = await session.run({ [session.inputNames[0]]: warmup });

      outputMap = {};
      for (const name of session.outputNames) {
        const size = results[name].data.length;
        if (size === 63 && !outputMap.landmarks) outputMap.landmarks = name;
        else if (size === 63) outputMap.worldLandmarks = name;
        else if (size === 1 && !outputMap.handFlag) outputMap.handFlag = name;
        else if (size === 1) outputMap.handedness = name;
      }

      self.postMessage({ type: 'ready' });
    } catch (err) {
      self.postMessage({ type: 'error', message: err.message });
    }
  }

  if (type === 'infer') {
    try {
      const { bitmap, rect, vw, vh } = e.data;

      // Crop + preprocess (all in this worker, no main thread involvement)
      const inv = cropAndPreprocess(bitmap, rect);
      bitmap.close(); // release the transferred bitmap

      if (!inv) {
        self.postMessage({ type: 'result', handFlag: 0, landmarks: null });
        return;
      }

      // Run inference using preallocated buffer
      const tensor = new ort.Tensor('float32', inputBuffer, [1, 224, 224, 3]);
      const results = await session.run({ [session.inputNames[0]]: tensor });

      const rawLandmarks = outputMap.landmarks ? results[outputMap.landmarks].data : null;
      const handFlag = outputMap.handFlag ? results[outputMap.handFlag].data[0] : 0;
      const handedness = outputMap.handedness ? results[outputMap.handedness].data[0] : 0;

      // Project landmarks back to video-normalized [0,1] space (do it here, not main thread)
      let projectedLandmarks = null;
      if (rawLandmarks && rawLandmarks.length === 63) {
        projectedLandmarks = new Float32Array(63);
        for (let i = 0; i < 21; i++) {
          const ox = rawLandmarks[i * 3];
          const oy = rawLandmarks[i * 3 + 1];
          const oz = rawLandmarks[i * 3 + 2];
          projectedLandmarks[i * 3]     = (inv.a * ox + inv.b * oy + inv.c) / vw;
          projectedLandmarks[i * 3 + 1] = (inv.d * ox + inv.e * oy + inv.f) / vh;
          projectedLandmarks[i * 3 + 2] = oz / 224;
        }
      }

      self.postMessage({
        type: 'result',
        handFlag,
        handedness,
        landmarks: projectedLandmarks ? projectedLandmarks.buffer : null,
      }, projectedLandmarks ? [projectedLandmarks.buffer] : []);
    } catch (err) {
      self.postMessage({ type: 'error', message: err.message });
    }
  }
};
