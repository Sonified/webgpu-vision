// Hand tracking pipeline: loads models, runs inference, returns detections/landmarks.
// Uses two Web Workers for true parallel landmark inference.

import * as ort from 'onnxruntime-web/webgpu';
import { generateAnchors, decodeDetections } from './anchors.js';
import { weightedNMS, detectionToRect } from './nms.js';
import { preprocessPalm, preprocessLandmark } from './preprocessing.js';

const PALM_MODEL_URL = '/models/palm_detection_lite.onnx';
const LANDMARK_MODEL_URL = '/models/hand_landmark_full.onnx';
const HAND_FLAG_THRESHOLD = 0.5;
const LANDMARK_SIZE = 224;

// Rate-limited logger
function makeLogger(intervalMs = 2000) {
  let lastLog = 0;
  return function(msg, ...args) {
    const now = performance.now();
    if (now - lastLog > intervalMs) {
      console.log(msg, ...args);
      lastLog = now;
    }
  };
}
const logPalm = makeLogger(2000);
const logSlot = makeLogger(2000);
const logLandmark = makeLogger(2000);

/**
 * Compute the 4 corner points of a rotated rectangle (in pixels).
 */
function rotatedRectPoints(cx, cy, w, h, rotation) {
  const b = Math.cos(rotation) * 0.5;
  const a = Math.sin(rotation) * 0.5;
  const p0x = cx - a * h - b * w, p0y = cy + b * h - a * w;
  const p1x = cx + a * h - b * w, p1y = cy - b * h - a * w;
  const p2x = 2 * cx - p0x,       p2y = 2 * cy - p0y;
  const p3x = 2 * cx - p1x,       p3y = 2 * cy - p1y;
  return [[p0x, p0y], [p1x, p1y], [p2x, p2y], [p3x, p3y]];
}

/**
 * Wraps a Web Worker for landmark inference with a promise-based API.
 */
class LandmarkWorker {
  constructor() {
    this.worker = new Worker(
      new URL('./landmark-worker.js', import.meta.url),
      { type: 'module' }
    );
    this.pendingResolve = null;
    this.worker.onmessage = (e) => this._onMessage(e);
  }

  init(modelUrl) {
    return new Promise((resolve, reject) => {
      this.worker.onmessage = (e) => {
        if (e.data.type === 'ready') {
          this.worker.onmessage = (ev) => this._onMessage(ev);
          resolve();
        } else if (e.data.type === 'error') {
          reject(new Error(e.data.message));
        }
      };
      this.worker.postMessage({ type: 'init', modelUrl });
    });
  }

  infer(inputFloat32) {
    return new Promise((resolve) => {
      this.pendingResolve = resolve;
      // Transfer the buffer (zero-copy)
      const buffer = inputFloat32.buffer.slice(0);
      this.worker.postMessage(
        { type: 'infer', inputBuffer: buffer, id: 0 },
        [buffer]
      );
    });
  }

  _onMessage(e) {
    if (e.data.type === 'result' && this.pendingResolve) {
      const resolve = this.pendingResolve;
      this.pendingResolve = null;
      resolve({
        landmarks: new Float32Array(e.data.landmarksBuffer),
        handFlag: e.data.handFlag,
        handedness: e.data.handedness,
      });
    } else if (e.data.type === 'error') {
      console.error('Worker error:', e.data.message);
      if (this.pendingResolve) {
        const resolve = this.pendingResolve;
        this.pendingResolve = null;
        resolve({ landmarks: new Float32Array(0), handFlag: 0, handedness: 0 });
      }
    }
  }
}

/**
 * One hand slot: owns its own warp canvas and worker.
 */
class HandSlot {
  constructor(index, worker) {
    this.index = index;
    this.worker = worker;
    this.active = false;
    this.rect = null;
    this.landmarks = null;
    this.warpCanvas = new OffscreenCanvas(LANDMARK_SIZE, LANDMARK_SIZE);
    this.warpCtx = this.warpCanvas.getContext('2d', { willReadFrequently: true });
    this._lastCrop = null;
  }

  cropRotatedRect(video, rect) {
    const ctx = this.warpCtx;
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
    const e = S * (dx1) * invDet;
    const c = -a * p1x - b * p1y;
    const f = -d * p1x - e * p1y;

    ctx.resetTransform();
    ctx.clearRect(0, 0, S, S);
    ctx.setTransform(a, d, b, e, c, f);
    ctx.drawImage(video, 0, 0);
    ctx.resetTransform();

    const imageData = ctx.getImageData(0, 0, S, S);

    const detM = a * e - b * d;
    const inverseTransform = {
      a: e / detM, b: -b / detM,
      c: -(e * c - b * f) / detM,
      d: -d / detM, e: a / detM,
      f: -(-d * c + a * f) / detM,
    };

    return { imageData, inverseTransform };
  }

  async runLandmark(video, vw, vh) {
    const crop = this.cropRotatedRect(video, this.rect);
    if (!crop) return { landmarks: [], handFlag: 0, handedness: 0 };

    const { imageData, inverseTransform } = crop;
    this._lastCrop = imageData;
    const data = preprocessLandmark(imageData);

    // Send to worker, get results back
    const result = await this.worker.infer(data);

    // Project landmarks back to video-normalized [0,1] space
    const projectedLandmarks = [];
    if (result.landmarks.length === 63) {
      const inv = inverseTransform;
      for (let i = 0; i < 21; i++) {
        const ox = result.landmarks[i * 3];
        const oy = result.landmarks[i * 3 + 1];
        const oz = result.landmarks[i * 3 + 2];
        const vx = inv.a * ox + inv.b * oy + inv.c;
        const vy = inv.d * ox + inv.e * oy + inv.f;
        projectedLandmarks.push({ x: vx / vw, y: vy / vh, z: oz / LANDMARK_SIZE });
      }
    }

    return {
      landmarks: projectedLandmarks,
      handFlag: result.handFlag,
      handedness: result.handedness,
    };
  }
}

export class HandTracker {
  constructor() {
    this.palmSession = null;
    this.anchors = null;
    this.workers = [new LandmarkWorker(), new LandmarkWorker()];
    this.slots = null; // created after workers init
    this.ready = false;
    this.running = false;
  }

  async init(onStatus) {
    onStatus?.('Generating anchors...');
    this.anchors = generateAnchors();

    onStatus?.('Loading palm detection model...');
    this.palmSession = await ort.InferenceSession.create(PALM_MODEL_URL, {
      executionProviders: ['webgpu'],
      graphOptimizationLevel: 'all',
    });

    // Warmup palm detection
    const warmPalm = new ort.Tensor('float32', new Float32Array(192 * 192 * 3), [1, 192, 192, 3]);
    await this.palmSession.run({ [this.palmSession.inputNames[0]]: warmPalm });

    // Initialize two landmark workers (sequentially to avoid WebGPU init conflicts,
    // but they'll run in TRUE parallel after init)
    onStatus?.('Loading landmark worker 0...');
    await this.workers[0].init(LANDMARK_MODEL_URL);
    onStatus?.('Loading landmark worker 1...');
    await this.workers[1].init(LANDMARK_MODEL_URL);

    this.slots = [
      new HandSlot(0, this.workers[0]),
      new HandSlot(1, this.workers[1]),
    ];

    console.log('Two landmark workers initialized -- true parallel inference enabled');

    this.ready = true;
    onStatus?.('Ready');
  }

  async runPalmDetection(video) {
    const { data, letterbox } = preprocessPalm(video);
    const input = new ort.Tensor('float32', data, [1, 192, 192, 3]);

    const results = await this.palmSession.run({
      [this.palmSession.inputNames[0]]: input,
    });

    let regressors, scores;
    for (const name of this.palmSession.outputNames) {
      const t = results[name];
      if (t.dims[t.dims.length - 1] === 18) regressors = t.data;
      else scores = t.data;
    }

    let detections = decodeDetections(regressors, scores, this.anchors);
    detections = weightedNMS(detections);

    if (detections.length > 0) {
      logPalm(`[palm] ${detections.length} detections`, detections.map(d =>
        `score=${d.score.toFixed(2)} cx=${d.cx.toFixed(3)} cy=${d.cy.toFixed(3)}`
      ));
    }
    return { detections, letterbox };
  }

  async processFrame(video) {
    if (!this.ready || this.running) return { hands: [] };
    this.running = true;

    const vw = video.videoWidth;
    const vh = video.videoHeight;

    try {
      // Palm detection: only when there are empty slots
      const emptySlots = this.slots.filter(s => !s.active);

      if (emptySlots.length > 0) {
        const { detections, letterbox } = await this.runPalmDetection(video);

        for (const det of detections) {
          if (emptySlots.length === 0) break;

          det.cx = (det.cx - letterbox.offsetX) / letterbox.scaleX;
          det.cy = (det.cy - letterbox.offsetY) / letterbox.scaleY;
          det.w = det.w / letterbox.scaleX;
          det.h = det.h / letterbox.scaleY;
          for (const kp of det.keypoints) {
            kp.x = (kp.x - letterbox.offsetX) / letterbox.scaleX;
            kp.y = (kp.y - letterbox.offsetY) / letterbox.scaleY;
          }

          if (det.cy < -0.1 || det.cy > 1.1 || det.cx < -0.1 || det.cx > 1.1) continue;

          const detPx = det.cx * vw;
          const detPy = det.cy * vh;
          const overlapsTracked = this.slots.some(s => {
            if (!s.active) return false;
            const dx = s.rect.cx - detPx;
            const dy = s.rect.cy - detPy;
            return Math.sqrt(dx * dx + dy * dy) < s.rect.w * 0.5;
          });
          if (overlapsTracked) continue;

          const rect = detectionToRect(det, vw, vh);
          const slot = emptySlots.shift();
          slot.active = true;
          slot.rect = rect;
          logSlot(`[new hand] slot ${slot.index} cx=${rect.cx.toFixed(0)} cy=${rect.cy.toFixed(0)}`);
        }
      }

      // TRUE PARALLEL: both hands infer simultaneously in separate workers
      const results = await Promise.all(this.slots.map(async (slot) => {
        if (!slot.active) return null;

        const result = await slot.runLandmark(video, vw, vh);

        if (result.handFlag > HAND_FLAG_THRESHOLD) {
          slot.landmarks = result.landmarks;
          slot.rect = this.landmarksToRect(result.landmarks, vw, vh);
          return { landmarks: result.landmarks, handedness: result.handedness };
        } else {
          slot.active = false;
          slot.landmarks = null;
          return null;
        }
      }));

      logLandmark(`[tracking] slots: ${this.slots.map(s => s.active ? '1' : '0').join(',')}`);

      return {
        hands: results.filter(Boolean),
        debug: {
          rects: this.slots.filter(s => s.rect).map(s => s.rect),
          lastCrop: this.slots[0]._lastCrop || this.slots[1]._lastCrop || null,
        },
      };
    } catch (err) {
      console.error('processFrame error:', err.message, err.stack);
      return { hands: [] };
    } finally {
      this.running = false;
    }
  }

  landmarksToRect(landmarks, imgW, imgH) {
    const wrist = landmarks[0];
    const indexMcp = landmarks[5];
    const middleMcp = landmarks[9];
    const ringMcp = landmarks[13];

    const tx = 0.25 * (indexMcp.x + ringMcp.x) + 0.5 * middleMcp.x;
    const ty = 0.25 * (indexMcp.y + ringMcp.y) + 0.5 * middleMcp.y;

    const rotation = Math.PI / 2 - Math.atan2(wrist.y - ty, tx - wrist.x);
    const angle = rotation - 2 * Math.PI * Math.floor((rotation + Math.PI) / (2 * Math.PI));

    const stableIds = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18];
    const pts = stableIds.map(i => [landmarks[i].x * imgW, landmarks[i].y * imgH]);

    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const [x, y] of pts) {
      minX = Math.min(minX, x); minY = Math.min(minY, y);
      maxX = Math.max(maxX, x); maxY = Math.max(maxY, y);
    }
    const acx = (minX + maxX) / 2;
    const acy = (minY + maxY) / 2;

    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    let rMinX = Infinity, rMinY = Infinity, rMaxX = -Infinity, rMaxY = -Infinity;
    for (const [x, y] of pts) {
      const dx = x - acx, dy = y - acy;
      const rx = dx * cos + dy * sin;
      const ry = -dx * sin + dy * cos;
      rMinX = Math.min(rMinX, rx); rMinY = Math.min(rMinY, ry);
      rMaxX = Math.max(rMaxX, rx); rMaxY = Math.max(rMaxY, ry);
    }

    const projCx = (rMinX + rMaxX) / 2;
    const projCy = (rMinY + rMaxY) / 2;
    const cx = cos * projCx - sin * projCy + acx;
    const cy = sin * projCx + cos * projCy + acy;

    const width = rMaxX - rMinX;
    const height = rMaxY - rMinY;
    const size = 2 * Math.max(width, height);

    const shiftCx = cx + 0.1 * height * sin;
    const shiftCy = cy - 0.1 * height * cos;

    return { cx: shiftCx, cy: shiftCy, w: size, h: size, angle };
  }
}
