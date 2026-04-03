// Anchor generation and detection decoding for BlazePalm (palm detection model).
// Reference: PINTO0309/hand-gesture-recognition-using-onnx, geaxgx/depthai_hand_tracker

const INPUT_SIZE = 192;
const NUM_ANCHORS = 2016;
const SCORE_THRESHOLD = 0.5;
const VALUES_PER_ANCHOR = 18; // bbox(4) + 7 keypoints * 2 coords

/**
 * Generate 2016 anchor positions for BlazePalm.
 * Returns Float32Array of (x_center, y_center) pairs, length 4032.
 */
export function generateAnchors() {
  const strides = [8, 16, 16, 16];
  const anchorsPerStride = [2, 2, 2, 2]; // 2 anchors per grid cell per layer

  const anchors = [];

  for (let layer = 0; layer < strides.length; layer++) {
    const stride = strides[layer];
    const gridSize = Math.ceil(INPUT_SIZE / stride);
    const count = anchorsPerStride[layer];

    for (let y = 0; y < gridSize; y++) {
      for (let x = 0; x < gridSize; x++) {
        const cx = (x + 0.5) / gridSize;
        const cy = (y + 0.5) / gridSize;
        for (let n = 0; n < count; n++) {
          anchors.push(cx, cy);
        }
      }
    }
  }

  const result = new Float32Array(anchors);

  if (result.length !== NUM_ANCHORS * 2) {
    throw new Error(
      `Anchor count mismatch: expected ${NUM_ANCHORS}, got ${result.length / 2}`
    );
  }

  return result;
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

/**
 * Decode raw model outputs into filtered detections.
 *
 * @param {Float32Array} regressors - Raw regressor output, shape (2016, 18)
 * @param {Float32Array} scores     - Raw score output, shape (2016, 1)
 * @param {Float32Array} anchors    - Anchor positions from generateAnchors()
 * @returns {Array<{cx, cy, w, h, score, keypoints}>} Filtered detections
 */
export function decodeDetections(regressors, scores, anchors) {
  const detections = [];

  for (let i = 0; i < NUM_ANCHORS; i++) {
    const score = sigmoid(scores[i]);
    if (score < SCORE_THRESHOLD) continue;

    const ax = anchors[i * 2];
    const ay = anchors[i * 2 + 1];
    const ri = i * VALUES_PER_ANCHOR;

    const cx = regressors[ri + 0] / INPUT_SIZE + ax;
    const cy = regressors[ri + 1] / INPUT_SIZE + ay;
    const w  = regressors[ri + 2] / INPUT_SIZE;
    const h  = regressors[ri + 3] / INPUT_SIZE;

    const keypoints = [];
    for (let k = 0; k < 7; k++) {
      keypoints.push({
        x: regressors[ri + 4 + k * 2]     / INPUT_SIZE + ax,
        y: regressors[ri + 4 + k * 2 + 1] / INPUT_SIZE + ay,
      });
    }

    detections.push({ cx, cy, w, h, score, keypoints });
  }

  return detections;
}
