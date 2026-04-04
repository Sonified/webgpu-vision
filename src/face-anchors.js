// Anchor generation and detection decoding for BlazeFace (face detection model).
// Same algorithm as BlazePalm but: 128x128 input, 896 anchors, 6 keypoints.

const INPUT_SIZE = 128;
const NUM_ANCHORS = 896;
const SCORE_THRESHOLD = 0.5;
const VALUES_PER_ANCHOR = 16; // bbox(4) + 6 keypoints * 2 coords

/**
 * Generate 896 anchor positions for BlazeFace.
 * Returns Float32Array of (x_center, y_center) pairs, length 1792.
 */
export function generateFaceAnchors() {
  const strides = [8, 16, 16, 16];
  const anchorsPerStride = [2, 2, 2, 2];

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
 * @param {Float32Array} regressors - Raw regressor output, shape (896, 16)
 * @param {Float32Array} scores     - Raw score output, shape (896, 1)
 * @param {Float32Array} anchors    - Anchor positions from generateFaceAnchors()
 * @returns {Array<{cx, cy, w, h, score, keypoints}>} Filtered detections
 */
export function decodeFaceDetections(regressors, scores, anchors) {
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

    // 6 keypoints: left eye, right eye, nose, mouth, left ear, right ear
    const keypoints = [];
    for (let k = 0; k < 6; k++) {
      keypoints.push({
        x: regressors[ri + 4 + k * 2]     / INPUT_SIZE + ax,
        y: regressors[ri + 4 + k * 2 + 1] / INPUT_SIZE + ay,
      });
    }

    detections.push({ cx, cy, w, h, score, keypoints });
  }

  return detections;
}
