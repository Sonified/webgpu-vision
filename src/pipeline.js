// Hand tracking pipeline: ALL inference in workers. Main thread is pure orchestration.
// - Palm detection: dedicated worker with GPU letterbox
// - Landmark inference: two parallel workers with GPU affine warp

import { detectionToRect } from './nms.js';
import { PALM_MODEL_URL, HAND_LANDMARK_URL as LANDMARK_MODEL_URL } from './model-urls.js';
import { workerUrlWithGates, registerWorkerForGateUpdates, log, makeLogger } from './log-gates.js';
// Tracking continues while the landmark model's handFlag stays above this.
// 0.5 was too strict -- fast motion or brief occlusion (hand-over-hand claps,
// quick swipes) dipped the flag below for a frame and killed the slot.
// 0.3 keeps tracking through transient drops; palm detection still gates
// the initial detection so false positives are rare.
let HAND_FLAG_THRESHOLD = 0.3;

const logPalm = makeLogger('tracking', 2000);
const logSlot = makeLogger('tracking', 2000);
const logLandmark = makeLogger('tracking', 2000);

/**
 * Wraps the palm detection worker.
 */
class PalmWorker {
  constructor() {
    this.worker = new Worker(
      workerUrlWithGates(new URL('./palm-worker.js', import.meta.url)),
      { type: 'module' }
    );
    registerWorkerForGateUpdates(this.worker);
    this.pendingResolve = null;
    this.worker.onmessage = (e) => this._onMessage(e);
    this.worker.onerror = (e) => console.error('[PalmWorker] uncaught worker error:', e.message, e);
  }

  init(modelUrl) {
    return new Promise((resolve, reject) => {
      this.worker.onmessage = (e) => {
        if (e.data.type === 'ready') {
          this.worker.onmessage = (ev) => this._onMessage(ev);
          resolve();
        } else if (e.data.type === 'error') {
          console.error('[PalmWorker] reported error:', e.data.message);
          reject(new Error(e.data.message));
        }
      };
      this.worker.onerror = (e) => {
        console.error('[PalmWorker] worker crashed:', e.message, e);
        reject(new Error(`Worker crashed: ${e.message}`));
      };
      this.worker.postMessage({ type: 'init', modelUrl });
    });
  }

  detect(frame) {
    return new Promise((resolve) => {
      this.pendingResolve = resolve;
      this.worker.postMessage({ type: 'detect', bitmap: frame, frame }, [frame]);
    });
  }

  _onMessage(e) {
    if (e.data.type === 'detections' && this.pendingResolve) {
      const resolve = this.pendingResolve;
      this.pendingResolve = null;
      resolve({ detections: e.data.detections, letterbox: e.data.letterbox, previewRGBA: e.data.previewRGBA });
    } else if (e.data.type === 'error') {
      console.error('Palm worker error:', e.data.message);
      if (this.pendingResolve) {
        this.pendingResolve({ detections: [], letterbox: {} });
        this.pendingResolve = null;
      }
    }
  }
}

/**
 * Wraps a landmark inference worker.
 */
class LandmarkWorker {
  constructor() {
    this.worker = new Worker(
      workerUrlWithGates(new URL('./landmark-worker-wgsl.js', import.meta.url)),
      { type: 'module' }
    );
    registerWorkerForGateUpdates(this.worker);
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

  infer(frame, rect, vw, vh) {
    return new Promise((resolve) => {
      this.pendingResolve = resolve;
      this.worker.postMessage(
        { type: 'infer', frame, rect, vw, vh },
        [frame]
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

      let worldLandmarks = [];
      if (e.data.worldLandmarks) {
        const flat = new Float32Array(e.data.worldLandmarks);
        for (let i = 0; i < 21; i++) {
          worldLandmarks.push({
            x: flat[i * 3],
            y: flat[i * 3 + 1],
            z: flat[i * 3 + 2],
          });
        }
      }

      resolve({
        landmarks,
        worldLandmarks,
        handFlag: e.data.handFlag,
        handedness: e.data.handedness,
      });
    } else if (e.data.type === 'error') {
      console.error('Landmark worker error:', e.data.message);
      if (this.pendingResolve) {
        this.pendingResolve({ landmarks: [], handFlag: 0, handedness: null });
        this.pendingResolve = null;
      }
    }
  }
}

// Palm centroid: mean of wrist + 3 palm MCPs. These four landmarks are the
// most spatially stable on a hand -- they move together as a rigid body while
// fingers flex. Used as the per-slot identity anchor for nearest-neighbor
// assignment between consecutive frames.
const PALM_IDS = [0, 5, 9, 17];
function palmCentroid(landmarks, vw, vh) {
  let cx = 0, cy = 0;
  for (const i of PALM_IDS) { cx += landmarks[i].x; cy += landmarks[i].y; }
  return { x: (cx / PALM_IDS.length) * vw, y: (cy / PALM_IDS.length) * vh };
}
function distSq(a, b) {
  const dx = a.x - b.x, dy = a.y - b.y;
  return dx * dx + dy * dy;
}

// How many consecutive failed-inference frames before we give up on a slot.
// Grace period lets the slot survive transient handFlag dips (fast motion,
// brief occlusion) without killing the visual.
const MAX_MISSES = 3;

export class HandTracker {
  constructor() {
    this.palmWorker = new PalmWorker();
    this.landmarkWorkers = [new LandmarkWorker(), new LandmarkWorker()];
    // Each slot carries the minimum state needed for identity tracking:
    //   - rect: ROI for next inference (derived from last-accepted landmarks)
    //   - landmarks: last-accepted landmarks (returned during miss frames too)
    //   - centroid: last-accepted palm centroid (identity anchor)
    //   - lastHandedness: last-seen classifier label (for handedness-aware UI)
    //   - missFrames: consecutive failed-inference count
    //   - lastCentroid: centroid just before the slot went inactive, kept so
    //     a returning detection can be routed back to its original slot.
    this.slots = [
      { index: 0, worker: this.landmarkWorkers[0], active: false, rect: null, landmarks: null, centroid: null, lastCentroid: null, lastHandedness: null, missFrames: 0 },
      { index: 1, worker: this.landmarkWorkers[1], active: false, rect: null, landmarks: null, centroid: null, lastCentroid: null, lastHandedness: null, missFrames: 0 },
    ];
    this.ready = false;
    this.running = false;
    this.palmDetecting = false;
    this.pendingDetections = null;
    // Tunable: pixel distance at which two slot inferences are considered
    // duplicates (both tracking the same physical hand). Smaller = only
    // near-exact centroid overlap triggers dedup; larger = more aggressive.
    this.dupDistPx = 25;
    this.handFlagThreshold = HAND_FLAG_THRESHOLD;
  }

  async init(onStatus) {
    // Init all three workers sequentially (WebGPU EP requires it)
    onStatus?.('Loading palm detection worker...');
    await this.palmWorker.init(PALM_MODEL_URL);

    onStatus?.('Loading landmark worker 0...');
    await this.landmarkWorkers[0].init(LANDMARK_MODEL_URL);
    onStatus?.('Loading landmark worker 1...');
    await this.landmarkWorkers[1].init(LANDMARK_MODEL_URL);

    log('lifecycle', '[lifecycle] All workers ready -- main thread is pure orchestration');
    this.ready = true;
    onStatus?.('Ready');
  }

  async processFrame(video) {
    if (!this.ready || this.running) return { hands: [] };
    this.running = true;

    const vw = video.videoWidth;
    const vh = video.videoHeight;

    try {
      // --- 1. Pending palm detections: route new hands into empty slots ---
      // Palm detection returns ALL hands it sees, even ones we're already
      // tracking. For each empty slot, pick the detection FARTHEST from any
      // active slot -- that's the one most likely to be the new hand (the
      // one we aren't tracking yet). This is critical when hands are close
      // together: naive overlap rejection would discard the legitimate new
      // hand detection just because it's near the tracked one.
      if (this.pendingDetections) {
        const { detections, letterbox } = this.pendingDetections;
        this.pendingDetections = null;

        // Normalize all detections first (coord mapping, frame-bounds filter)
        const norm = [];
        for (const det of detections) {
          det.cx = (det.cx - letterbox.offsetX) / letterbox.scaleX;
          det.cy = (det.cy - letterbox.offsetY) / letterbox.scaleY;
          det.w = det.w / letterbox.scaleX;
          det.h = det.h / letterbox.scaleY;
          for (const kp of det.keypoints) {
            kp.x = (kp.x - letterbox.offsetX) / letterbox.scaleX;
            kp.y = (kp.y - letterbox.offsetY) / letterbox.scaleY;
          }
          if (det.cy < -0.1 || det.cy > 1.1 || det.cx < -0.1 || det.cx > 1.1) continue;
          norm.push({ det, point: { x: det.cx * vw, y: det.cy * vh } });
        }

        // For each detection, compute its distance to the nearest active slot's
        // centroid. Detections FAR from all active slots are strong candidates
        // for a new hand; detections CLOSE to an active slot are likely the
        // same hand we're already tracking (duplicate palm detection).
        for (const n of norm) {
          n.minDistSqToActive = Infinity;
          for (const s of this.slots) {
            if (!s.active || !s.centroid) continue;
            const d = distSq(s.centroid, n.point);
            if (d < n.minDistSqToActive) n.minDistSqToActive = d;
          }
        }
        // Sort so farthest-from-active is first (most likely a new hand).
        norm.sort((a, b) => b.minDistSqToActive - a.minDistSqToActive);

        // If both slots are active and we have 2+ detections, we fired palm
        // because hands overlap. Re-anchor each slot's ROI to the palm
        // detection closest to that slot's current centroid. This gives
        // landmark inference a fresh, tight ROI instead of one that has
        // drifted to cover both hands.
        const bothActive = this.slots[0].active && this.slots[1].active;
        if (bothActive && norm.length >= 2) {
          // For each slot, find the detection closest to its current centroid.
          // Then if the two slots claimed the same detection, a spurious
          // situation -- skip re-anchor this frame rather than risk misrouting.
          const s0c = this.slots[0].centroid, s1c = this.slots[1].centroid;
          let n0 = null, n1 = null, n0d = Infinity, n1d = Infinity;
          for (const n of norm) {
            const d0 = distSq(n.point, s0c);
            const d1 = distSq(n.point, s1c);
            if (d0 < n0d) { n0d = d0; n0 = n; }
            if (d1 < n1d) { n1d = d1; n1 = n; }
          }
          // Only re-anchor if each slot picked a DIFFERENT detection and
          // both picks are reasonably close (<100px, rules out corner
          // false positives).
          const REASONABLE_PX_SQ = 100 * 100;
          if (n0 && n1 && n0 !== n1 && n0d < REASONABLE_PX_SQ && n1d < REASONABLE_PX_SQ) {
            this.slots[0].rect = detectionToRect(n0.det, vw, vh);
            this.slots[0].centroid = { x: this.slots[0].rect.cx, y: this.slots[0].rect.cy };
            this.slots[1].rect = detectionToRect(n1.det, vw, vh);
            this.slots[1].centroid = { x: this.slots[1].rect.cx, y: this.slots[1].rect.cy };
            logSlot(`[reanchor] s0=(${this.slots[0].centroid.x.toFixed(0)},${this.slots[0].centroid.y.toFixed(0)}) s1=(${this.slots[1].centroid.x.toFixed(0)},${this.slots[1].centroid.y.toFixed(0)})`);
          }
          // Fall through: empty-slot fill loop below does nothing when all slots are active.
        }

        const assignedPoints = [];
        for (const n of norm) {
          const emptySlots = this.slots.filter(s => !s.active);
          if (emptySlots.length === 0) break;

          let nearAssigned = false;
          for (const p of assignedPoints) {
            if (distSq(p, n.point) < 80 * 80) { nearAssigned = true; break; }
          }
          if (nearAssigned) continue;

          let closeThreshSq = 40 * 40;
          for (const s of this.slots) {
            if (!s.active || !s.rect) continue;
            const excl = s.rect.w * 0.75;
            if (excl * excl > closeThreshSq) closeThreshSq = excl * excl;
          }
          if (n.minDistSqToActive < closeThreshSq) {
            console.log(`[palm-reject] det at (${n.point.x.toFixed(0)},${n.point.y.toFixed(0)}) too close: dist=${Math.sqrt(n.minDistSqToActive).toFixed(0)}px < thresh=${Math.sqrt(closeThreshSq).toFixed(0)}px`);
            continue;
          }

          let bestSlot = emptySlots[0];
          let bestDist = Infinity;
          for (const s of emptySlots) {
            const ref = s.lastCentroid || s.centroid;
            if (!ref) continue;
            const d = distSq(ref, n.point);
            if (d < bestDist) { bestDist = d; bestSlot = s; }
          }

          const rect = detectionToRect(n.det, vw, vh);
          if (rect.cx < 0 || rect.cx > vw || rect.cy < 0 || rect.cy > vh) continue;
          bestSlot.active = true;
          bestSlot.rect = rect;
          bestSlot.centroid = { x: rect.cx, y: rect.cy };
          bestSlot.missFrames = 0;
          assignedPoints.push(n.point);
          logSlot(`[new hand] slot ${bestSlot.index} cx=${rect.cx.toFixed(0)} cy=${rect.cy.toFixed(0)}`);
        }
      }

      // --- 2. Fire palm detection ---
      // Two scenarios:
      //   a) An empty slot needs filling.
      //   b) Both slots active AND centroids are GENUINELY overlapping
      //      (< 80px, roughly one palm width). In this overlap case the
      //      landmark model can't distinguish the two hands -- one slot's
      //      ROI covers the other hand. Palm detection sees both hands as
      //      separate boxes and lets us re-anchor each slot correctly.
      const hasEmptySlots = this.slots.some(s => !s.active);
      const s0 = this.slots[0], s1 = this.slots[1];
      const slotsOverlapping = s0.active && s1.active && s0.centroid && s1.centroid
        && distSq(s0.centroid, s1.centroid) < 80 * 80;
      if ((hasEmptySlots || slotsOverlapping) && !this.palmDetecting) {
        this.palmDetecting = true;
        this._palmAttempts = (this._palmAttempts || 0) + 1;
        const attemptNum = this._palmAttempts;
        const t0 = performance.now();
        const frame = new VideoFrame(video);
        this.palmWorker.detect(frame).then(result => {
          this.palmDetecting = false;
          const ms = (performance.now() - t0).toFixed(1);
          const empty = this.slots.filter(s => !s.active).length;
          console.log(`[palm-search] attempt #${attemptNum}: ${result.detections.length} det in ${ms}ms, ${empty} empty slot(s)`);
          if (result.previewRGBA) this._lastPreview = result.previewRGBA;
          if (result.detections.length > 0) {
            logPalm(`[palm] ${result.detections.length} detections`);
            this.pendingDetections = result;
          }
        }).catch(() => { this.palmDetecting = false; });
      }

      // --- 3. Run landmark inference on every active slot ---
      // Snapshot each slot's previous centroid BEFORE inference; used below
      // for nearest-neighbor assignment to handle cross-slot identity swaps.
      const priorCentroids = this.slots.map(s => s.centroid ? { ...s.centroid } : null);

      const rawResults = await Promise.all(this.slots.map(async (slot) => {
        if (!slot.active) return { slotIdx: slot.index, result: null };
        const frame = new VideoFrame(video);
        const result = await slot.worker.infer(frame, slot.rect, vw, vh);
        return { slotIdx: slot.index, result };
      }));

      // --- 4. Keep only strong inferences (above threshold) ---
      const strong = [];
      for (const r of rawResults) {
        if (!r.result) continue;
        if (r.result.handFlag <= this.handFlagThreshold) continue;
        strong.push({
          slotIdx: r.slotIdx,
          handFlag: r.result.handFlag,
          handedness: r.result.handedness,
          landmarks: r.result.landmarks,
          worldLandmarks: r.result.worldLandmarks,
          centroid: palmCentroid(r.result.landmarks, vw, vh),
        });
      }

      // --- 5. Assignment: map each strong result to a slot by nearest prior centroid ---
      // Handles the prayer-hands / fast-clap case where the landmark model
      // running on slot A's ROI may return hand B's landmarks (and vice
      // versa). Instead of rejecting, we re-route: the landmarks go to
      // whichever slot's previous centroid is closest to the new centroid.
      //
      // Also handles the DUPLICATE case: both slots' inferences return the
      // same physical hand (common when hands overlap). We detect this by
      // centroid proximity between the two results and keep only one --
      // otherwise both slots end up tracking the same hand forever.
      const assigned = [null, null]; // assigned[slotIdx] = strong entry
      const DUP_DIST_SQ = this.dupDistPx * this.dupDistPx;

      if (strong.length === 1) {
        const s = strong[0];
        const p0 = priorCentroids[0], p1 = priorCentroids[1];
        if (p0 && p1) {
          assigned[distSq(s.centroid, p0) <= distSq(s.centroid, p1) ? 0 : 1] = s;
        } else if (p0) assigned[0] = s;
        else if (p1) assigned[1] = s;
        else assigned[s.slotIdx] = s; // cold start
      } else if (strong.length === 2) {
        const [r0, r1] = strong;
        const p0 = priorCentroids[0], p1 = priorCentroids[1];

        // Duplicate check: if both result centroids are close to each other,
        // both inferences locked onto the same physical hand. Keep the one
        // whose centroid better matches its original slot's prior centroid
        // and let the other slot enter miss state.
        if (distSq(r0.centroid, r1.centroid) < DUP_DIST_SQ) {
          log('tracking', `[tracking] duplicate: s0=(${r0.centroid.x.toFixed(0)},${r0.centroid.y.toFixed(0)}) s1=(${r1.centroid.x.toFixed(0)},${r1.centroid.y.toFixed(0)}) dist=${Math.sqrt(distSq(r0.centroid, r1.centroid)).toFixed(0)}`);
          let winner = r0, winnerSlot = r0.slotIdx;
          if (p0 && p1) {
            // For each result, measure how well it matches its originating slot
            const r0Fit = r0.slotIdx === 0 ? distSq(r0.centroid, p0) : distSq(r0.centroid, p1);
            const r1Fit = r1.slotIdx === 0 ? distSq(r1.centroid, p0) : distSq(r1.centroid, p1);
            if (r1Fit < r0Fit) { winner = r1; winnerSlot = r1.slotIdx; }
            // Route winner to the closest prior centroid
            winnerSlot = distSq(winner.centroid, p0) <= distSq(winner.centroid, p1) ? 0 : 1;
          }
          assigned[winnerSlot] = winner;
        } else if (p0 && p1) {
          const cIdentity = distSq(r0.centroid, p0) + distSq(r1.centroid, p1);
          const cSwap = distSq(r0.centroid, p1) + distSq(r1.centroid, p0);
          if (cIdentity <= cSwap) { assigned[0] = r0; assigned[1] = r1; }
          else { assigned[0] = r1; assigned[1] = r0; }
        } else {
          // Cold start: trust original slot assignment
          assigned[r0.slotIdx] = r0;
          assigned[r1.slotIdx] = r1;
        }
      }

      // --- 6. Apply assignments and update slot state ---
      for (let i = 0; i < 2; i++) {
        const slot = this.slots[i];
        if (!slot.active) continue;
        const a = assigned[i];
        if (a) {
          slot.missFrames = 0;
          slot.landmarks = a.landmarks;
          slot.worldLandmarks = a.worldLandmarks;
          slot.centroid = a.centroid;
          slot.lastHandedness = a.handedness;
          slot.lastHandFlag = a.handFlag;
          slot.rect = this.landmarksToRect(a.landmarks, vw, vh);
        } else {
          // No strong inference assigned to this slot this frame. Grace
          // period: hold last landmarks for up to MAX_MISSES frames so brief
          // dips (fast motion, model confused for 1 frame) don't kill the
          // visual. After that, drop the slot and let palm detection pick
          // up the hand again if it's still there.
          slot.missFrames++;
          if (slot.missFrames >= MAX_MISSES) {
            log('tracking', `[tracking] slot ${i} dropped after ${slot.missFrames} misses`);
            slot.lastCentroid = slot.centroid;
            slot.active = false;
            slot.landmarks = null;
            slot.worldLandmarks = null;
            slot.centroid = null;
            slot.missFrames = 0;
          }
        }
      }

      // --- 7. Build return value in stable slot order ---
      // Returning a fixed 2-element array where index = slot identity means
      // the caller (demo) can trust hands[0] and hands[1] to stay on the
      // same physical hand across frames. Null = that slot is empty.
      const hands = this.slots.map(s => {
        if (!s.landmarks) return null;
        return { landmarks: s.landmarks, worldLandmarks: s.worldLandmarks || [], handedness: s.lastHandedness, handFlag: s.lastHandFlag || 0 };
      });

      logLandmark(`[tracking] slots: ${this.slots.map(s => s.active ? String(s.index) : '_').join(',')}`);

      const result = {
        hands,
        stableIdentity: true,
        debug: { rects: this.slots.filter(s => s.rect).map(s => s.rect) },
      };
      if (this._lastPreview) {
        result.palmPreview = this._lastPreview;
        this._lastPreview = null;
      }
      return result;
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
