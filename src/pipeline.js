// Hand tracking pipeline: palm detection on main thread, landmark inference in workers.
// Workers receive ImageBitmaps and do crop + preprocess + inference -- main thread stays light.

import * as ort from 'onnxruntime-web/webgpu';
import { generateAnchors, decodeDetections } from './anchors.js';
import { weightedNMS, detectionToRect } from './nms.js';
import { preprocessPalm } from './preprocessing.js';

const PALM_MODEL_URL = '/models/palm_detection_lite.onnx';
const LANDMARK_MODEL_URL = '/models/hand_landmark_full.onnx';
const HAND_FLAG_THRESHOLD = 0.5;

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
 * Wraps a Web Worker with a promise-based inference API.
 * Sends ImageBitmap + rect, receives projected landmarks.
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

  infer(bitmap, rect, vw, vh) {
    return new Promise((resolve) => {
      this.pendingResolve = resolve;
      this.worker.postMessage(
        { type: 'infer', bitmap, rect, vw, vh },
        [bitmap] // transfer bitmap (zero-copy)
      );
    });
  }

  _onMessage(e) {
    if (e.data.type === 'result' && this.pendingResolve) {
      const resolve = this.pendingResolve;
      this.pendingResolve = null;

      let landmarks = [];
      if (e.data.landmarks) {
        const flat = new Float32Array(e.data.landmarks);
        for (let i = 0; i < 21; i++) {
          landmarks.push({
            x: flat[i * 3],
            y: flat[i * 3 + 1],
            z: flat[i * 3 + 2],
          });
        }
      }

      resolve({
        landmarks,
        handFlag: e.data.handFlag,
        handedness: e.data.handedness,
      });
    } else if (e.data.type === 'error') {
      console.error('Worker error:', e.data.message);
      if (this.pendingResolve) {
        const resolve = this.pendingResolve;
        this.pendingResolve = null;
        resolve({ landmarks: [], handFlag: 0, handedness: 0 });
      }
    }
  }
}

export class HandTracker {
  constructor() {
    this.palmSession = null;
    this.anchors = null;
    this.workers = [new LandmarkWorker(), new LandmarkWorker()];
    this.slots = [
      { index: 0, worker: this.workers[0], active: false, rect: null, landmarks: null },
      { index: 1, worker: this.workers[1], active: false, rect: null, landmarks: null },
    ];
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

    const warmPalm = new ort.Tensor('float32', new Float32Array(192 * 192 * 3), [1, 192, 192, 3]);
    await this.palmSession.run({ [this.palmSession.inputNames[0]]: warmPalm });

    onStatus?.('Loading landmark worker 0...');
    await this.workers[0].init(LANDMARK_MODEL_URL);
    onStatus?.('Loading landmark worker 1...');
    await this.workers[1].init(LANDMARK_MODEL_URL);

    console.log('Two landmark workers ready -- true parallel inference');
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

      // TRUE PARALLEL: create ImageBitmaps and send to workers simultaneously
      const results = await Promise.all(this.slots.map(async (slot) => {
        if (!slot.active) return null;

        // Create a bitmap from the current video frame (fast, GPU-backed)
        const bitmap = await createImageBitmap(video);

        // Worker does: crop + preprocess + inference + projection
        const result = await slot.worker.infer(bitmap, slot.rect, vw, vh);

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
