// Hand tracking pipeline: loads models, runs inference, returns detections/landmarks.

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

// Reusable canvas for affine warp
let warpCanvas, warpCtx;
function getWarpCanvas() {
  if (!warpCanvas) {
    warpCanvas = new OffscreenCanvas(LANDMARK_SIZE, LANDMARK_SIZE);
    warpCtx = warpCanvas.getContext('2d', { willReadFrequently: true });
  }
  return warpCtx;
}

/**
 * Compute the 4 corner points of a rotated rectangle (in pixels).
 * Matches mediapipe_utils.py rotated_rect_to_points().
 */
function rotatedRectPoints(cx, cy, w, h, rotation) {
  const b = Math.cos(rotation) * 0.5;
  const a = Math.sin(rotation) * 0.5;
  // Match reference: p2 = opposite of p0, p3 = opposite of p1
  const p0x = cx - a * h - b * w, p0y = cy + b * h - a * w; // bottom-left
  const p1x = cx + a * h - b * w, p1y = cy - b * h - a * w; // top-left
  const p2x = 2 * cx - p0x,       p2y = 2 * cy - p0y;       // top-right
  const p3x = 2 * cx - p1x,       p3y = 2 * cy - p1y;       // bottom-right
  return [[p0x, p0y], [p1x, p1y], [p2x, p2y], [p3x, p3y]];
}

export class HandTracker {
  constructor() {
    this.palmSession = null;
    this.landmarkSession = null;
    this.anchors = null;
    this.handSlots = [
      { active: false, rect: null, landmarks: null },
      { active: false, rect: null, landmarks: null },
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

    onStatus?.('Loading hand landmark model...');
    this.landmarkSession = await ort.InferenceSession.create(LANDMARK_MODEL_URL, {
      executionProviders: ['webgpu'],
      graphOptimizationLevel: 'all',
    });

    console.log('Palm inputs:', this.palmSession.inputNames, 'outputs:', this.palmSession.outputNames);
    console.log('Landmark inputs:', this.landmarkSession.inputNames, 'outputs:', this.landmarkSession.outputNames);

    onStatus?.('Warming up WebGPU shaders...');
    const warmPalm = new ort.Tensor('float32', new Float32Array(192 * 192 * 3), [1, 192, 192, 3]);
    await this.palmSession.run({ [this.palmSession.inputNames[0]]: warmPalm });

    const warmLand = new ort.Tensor('float32', new Float32Array(224 * 224 * 3), [1, 224, 224, 3]);
    const warmResults = await this.landmarkSession.run({ [this.landmarkSession.inputNames[0]]: warmLand });

    // Map landmark outputs by shape
    this.landmarkOutputMap = {};
    for (const name of this.landmarkSession.outputNames) {
      const t = warmResults[name];
      const size = t.data.length;
      console.log(`  Landmark output "${name}": [${t.dims}]`);
      if (size === 63 && !this.landmarkOutputMap.landmarks) {
        this.landmarkOutputMap.landmarks = name;
      } else if (size === 63) {
        this.landmarkOutputMap.worldLandmarks = name;
      } else if (size === 1 && !this.landmarkOutputMap.handFlag) {
        this.landmarkOutputMap.handFlag = name;
      } else if (size === 1) {
        this.landmarkOutputMap.handedness = name;
      }
    }
    console.log('Landmark output map:', this.landmarkOutputMap);

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

  /**
   * Crop a rotated rectangle from the video using the 3-point affine approach
   * from mediapipe_utils.py warp_rect_img().
   *
   * rect has {cx, cy, w, h, angle} in PIXEL coordinates.
   */
  cropRotatedRect(video, rect) {
    const ctx = getWarpCanvas();
    const S = LANDMARK_SIZE;

    // Get the 4 corners of the rotated rect in pixel space
    const pts = rotatedRectPoints(rect.cx, rect.cy, rect.w, rect.h, rect.angle);
    // pts: [p0(BL), p1(TL), p2(BR), p3(TR)]
    // Reference uses pts[1:] = [p1, p2, p3] mapped to [(0,0), (S,0), (S,S)]
    // Affine warp: p1->(0,0), p2->(S,0), p3->(S,S)
    // Affine: [a b c; d e f] where output = M * input
    // (0,0) = M * p1,  (S,0) = M * p2,  (S,S) = M * p3

    const [p1x, p1y] = pts[1];
    const [p2x, p2y] = pts[2];
    const [p3x, p3y] = pts[3];

    // Solve for affine matrix:
    // [S, 0] = [a,b] * [p2x-p1x, p2y-p1y]^T + [0,0] => after subtracting p1
    // [S, S] = [a,b] * [p3x-p1x, p3y-p1y]^T
    const dx1 = p2x - p1x, dy1 = p2y - p1y;
    const dx2 = p3x - p1x, dy2 = p3y - p1y;
    const det = dx1 * dy2 - dx2 * dy1;

    if (Math.abs(det) < 1e-6) return null;

    const invDet = 1 / det;
    // Solve: a*dx1 + b*dy1 = S, a*dx2 + b*dy2 = S  (x-row: both p2,p3 map to x=S)
    //        d*dx1 + e*dy1 = 0, d*dx2 + e*dy2 = S  (y-row: p2->y=0, p3->y=S)
    const a = S * (dy2 - dy1) * invDet;
    const b = S * (dx1 - dx2) * invDet;
    const d = S * (-dy1) * invDet;
    const e = S * (dx1) * invDet;

    // Translation: dst = M * (src - p1)
    // c = -a*p1x - b*p1y,  f = -d*p1x - e*p1y
    const c = -a * p1x - b * p1y;
    const f = -d * p1x - e * p1y;

    ctx.resetTransform();
    ctx.clearRect(0, 0, S, S);
    // setTransform(a, d, b, e, c, f) -- note canvas param order: (a,b,c,d,e,f) = (m11,m12,m21,m22,dx,dy)
    ctx.setTransform(a, d, b, e, c, f);
    ctx.drawImage(video, 0, 0);
    ctx.resetTransform();

    const imageData = ctx.getImageData(0, 0, S, S);

    // Inverse affine for projecting landmarks back to video pixel space:
    // src = M_inv * dst, where M_inv = [dy2,-dx2; -dy1,dx1] * S / det ... actually easier:
    // dst_norm [0,1] -> src pixel: just reverse the affine
    // For landmark at (u,v) in [0,1] of 224x224 space:
    //   px_in_224 = u * S, py_in_224 = v * S
    //   video_x = (px_in_224 - c) / a ... no, need proper inverse
    // Inverse: video = M^-1 * output_px
    // M = [[a,b,c],[d,e,f]]
    // M^-1: det_m = a*e - b*d
    const detM = a * e - b * d;
    const invA = e / detM, invB = -b / detM;
    const invD = -d / detM, invE = a / detM;
    const invC = -(invA * c + invB * f);
    const invF = -(invD * c + invE * f);

    // inverseTransform maps output pixel (ox, oy) -> video pixel (vx, vy)
    // Then we normalize by video dimensions in runLandmarkModel
    const inverseTransform = { a: invA, b: invB, c: invC, d: invD, e: invE, f: invF };

    return { imageData, inverseTransform };
  }

  async runLandmarkModel(video, rect) {
    const crop = this.cropRotatedRect(video, rect);
    if (!crop) return { landmarks: [], handFlag: 0, handedness: 0 };

    const { imageData, inverseTransform } = crop;
    const data = preprocessLandmark(imageData);
    const input = new ort.Tensor('float32', data, [1, 224, 224, 3]);

    const results = await this.landmarkSession.run({
      [this.landmarkSession.inputNames[0]]: input,
    });

    const map = this.landmarkOutputMap;
    const landmarks = map.landmarks ? results[map.landmarks].data : null;
    const handFlag = map.handFlag ? results[map.handFlag].data[0] : 0;
    const handedness = map.handedness ? results[map.handedness].data[0] : 0;

    // Save crop for debug visualization
    this._lastCrop = imageData;

    const vw = video.videoWidth;
    const vh = video.videoHeight;

    // Project landmarks back to video-normalized [0,1] space
    const projectedLandmarks = [];
    if (landmarks) {
      const inv = inverseTransform;
      for (let i = 0; i < 21; i++) {
        // Landmark coords are in 224x224 pixel space
        const ox = landmarks[i * 3];
        const oy = landmarks[i * 3 + 1];
        const oz = landmarks[i * 3 + 2];

        // Apply inverse affine to get video pixel coords
        const vx = inv.a * ox + inv.b * oy + inv.c;
        const vy = inv.d * ox + inv.e * oy + inv.f;

        // Normalize to [0,1]
        projectedLandmarks.push({ x: vx / vw, y: vy / vh, z: oz / LANDMARK_SIZE });
      }
    }

    logLandmark(`[landmark] handFlag=${handFlag.toFixed(3)} handedness=${handedness.toFixed(3)}`);

    return { landmarks: projectedLandmarks, handFlag, handedness };
  }

  async processFrame(video) {
    if (!this.ready || this.running) return { hands: [] };
    this.running = true;

    const vw = video.videoWidth;
    const vh = video.videoHeight;

    try {
      // Run palm detection when there are empty slots (looking for new hands).
      // Never touch slots that are already tracking.
      const emptySlots = this.handSlots.filter(s => !s.active);

      if (emptySlots.length > 0) {
        const { detections, letterbox } = await this.runPalmDetection(video);

        for (const det of detections) {
          if (emptySlots.length === 0) break;

          // Undo letterbox
          det.cx = (det.cx - letterbox.offsetX) / letterbox.scaleX;
          det.cy = (det.cy - letterbox.offsetY) / letterbox.scaleY;
          det.w = det.w / letterbox.scaleX;
          det.h = det.h / letterbox.scaleY;
          for (const kp of det.keypoints) {
            kp.x = (kp.x - letterbox.offsetX) / letterbox.scaleX;
            kp.y = (kp.y - letterbox.offsetY) / letterbox.scaleY;
          }

          if (det.cy < -0.1 || det.cy > 1.1 || det.cx < -0.1 || det.cx > 1.1) continue;

          // Skip if this detection overlaps an already-tracked hand
          const detPx = det.cx * vw;
          const detPy = det.cy * vh;
          const overlapsTracked = this.handSlots.some(s => {
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
          logSlot(`[new hand] cx=${rect.cx.toFixed(0)} cy=${rect.cy.toFixed(0)} size=${rect.w.toFixed(0)}`);
        }
      }

      // Run landmark model sequentially
      const results = [];
      for (const slot of this.handSlots) {
        if (!slot.active) { results.push(null); continue; }

        const result = await this.runLandmarkModel(video, slot.rect);

        if (result.handFlag > HAND_FLAG_THRESHOLD) {
          slot.landmarks = result.landmarks;
          slot.rect = this.landmarksToRect(result.landmarks, vw, vh);
          results.push({ landmarks: result.landmarks, handedness: result.handedness });
        } else {
          slot.active = false;
          slot.landmarks = null;
          results.push(null);
        }
      }
      this.framesSincePalmDetect++;

      // Return debug info along with hands
      return {
        hands: results.filter(Boolean),
        debug: {
          rects: this.handSlots.filter(s => s.rect).map(s => s.rect),
          lastCrop: this._lastCrop || null,
        },
      };
    } catch (err) {
      console.error('processFrame error:', err);
      return { hands: [] };
    } finally {
      this.running = false;
    }
  }

  landmarksToRect(landmarks, imgW, imgH) {
    // Ported from geaxgx mediapipe_utils.py hand_landmarks_to_rect()

    // Rotation from wrist to averaged MCPs (more stable than single point)
    const wrist = landmarks[0];
    const indexMcp = landmarks[5];
    const middleMcp = landmarks[9];
    const ringMcp = landmarks[13];

    // Target point: weighted average of MCPs (middle finger weighted 2x)
    const tx = 0.25 * (indexMcp.x + ringMcp.x) + 0.5 * middleMcp.x;
    const ty = 0.25 * (indexMcp.y + ringMcp.y) + 0.5 * middleMcp.y;

    const rotation = Math.PI / 2 - Math.atan2(wrist.y - ty, tx - wrist.x);
    const angle = rotation - 2 * Math.PI * Math.floor((rotation + Math.PI) / (2 * Math.PI));

    // Only use stable palm/MCP landmarks (no fingertips!)
    const stableIds = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18];
    const pts = stableIds.map(i => [landmarks[i].x * imgW, landmarks[i].y * imgH]);

    // Axis-aligned center of stable points
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const [x, y] of pts) {
      minX = Math.min(minX, x); minY = Math.min(minY, y);
      maxX = Math.max(maxX, x); maxY = Math.max(maxY, y);
    }
    const acx = (minX + maxX) / 2;
    const acy = (minY + maxY) / 2;

    // Project points into rotated space, find bounds there
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

    // Center in rotated space, then back to pixel space
    const projCx = (rMinX + rMaxX) / 2;
    const projCy = (rMinY + rMaxY) / 2;
    const cx = cos * projCx - sin * projCy + acx;
    const cy = sin * projCx + cos * projCy + acy;

    // Square bounding box from longer side, 2x expansion
    const width = rMaxX - rMinX;
    const height = rMaxY - rMinY;
    const size = 2 * Math.max(width, height);

    // Shift center slightly along rotation axis (0.1 * height)
    const shiftCx = cx + 0.1 * height * sin;
    const shiftCy = cy - 0.1 * height * cos;

    return { cx: shiftCx, cy: shiftCy, w: size, h: size, angle };
  }
}
