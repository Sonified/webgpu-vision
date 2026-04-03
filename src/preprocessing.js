// Preprocessing for BlazePalm and Hand Landmark models.
// Uses OffscreenCanvas for GPU-accelerated resizing.

const PALM_SIZE = 192;
const LANDMARK_SIZE = 224;

// Reusable canvases -- created once, reused every frame.
let palmCanvas, palmCtx;
let landmarkCanvas, landmarkCtx;

function getPalmCanvas() {
  if (!palmCanvas) {
    palmCanvas = new OffscreenCanvas(PALM_SIZE, PALM_SIZE);
    palmCtx = palmCanvas.getContext('2d', { willReadFrequently: true });
  }
  return { canvas: palmCanvas, ctx: palmCtx };
}

function getLandmarkCanvas() {
  if (!landmarkCanvas) {
    landmarkCanvas = new OffscreenCanvas(LANDMARK_SIZE, LANDMARK_SIZE);
    landmarkCtx = landmarkCanvas.getContext('2d', { willReadFrequently: true });
  }
  return { canvas: landmarkCanvas, ctx: landmarkCtx };
}

/**
 * Preprocess a video frame for BlazePalm (palm detection).
 *
 * Steps:
 *   1. Letterbox pad the source to a square (preserving aspect ratio)
 *   2. Resize to 192x192 via OffscreenCanvas
 *   3. Normalize pixels to [-1, 1]: value = 2 * (pixel / 255) - 1
 *
 * @param {VideoFrame|HTMLVideoElement|HTMLCanvasElement|OffscreenCanvas|ImageBitmap} source
 * @returns {{ data: Float32Array, letterbox: { scaleX: number, scaleY: number, offsetX: number, offsetY: number } }}
 *   data is NHWC Float32Array of shape (1, 192, 192, 3).
 *   letterbox contains the transform info needed to map detections back to original coords.
 */
export function preprocessPalm(source) {
  const srcW = source.videoWidth || source.width;
  const srcH = source.videoHeight || source.height;
  const { ctx } = getPalmCanvas();

  // Letterbox: fit into a square, padding the shorter dimension.
  const scale = PALM_SIZE / Math.max(srcW, srcH);
  const dstW = Math.round(srcW * scale);
  const dstH = Math.round(srcH * scale);
  const offsetX = (PALM_SIZE - dstW) / 2;
  const offsetY = (PALM_SIZE - dstH) / 2;

  // Black fill for letterbox bars, then draw scaled source.
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, PALM_SIZE, PALM_SIZE);
  ctx.drawImage(source, offsetX, offsetY, dstW, dstH);

  const imageData = ctx.getImageData(0, 0, PALM_SIZE, PALM_SIZE);
  const rgba = imageData.data; // Uint8ClampedArray, length = 192*192*4

  const pixelCount = PALM_SIZE * PALM_SIZE;
  const data = new Float32Array(pixelCount * 3);

  for (let i = 0; i < pixelCount; i++) {
    const ri = i * 4;
    const oi = i * 3;
    data[oi]     = rgba[ri]     / 255; // R
    data[oi + 1] = rgba[ri + 1] / 255; // G
    data[oi + 2] = rgba[ri + 2] / 255; // B
  }

  return {
    data,
    letterbox: {
      scaleX: dstW / PALM_SIZE,
      scaleY: dstH / PALM_SIZE,
      offsetX: offsetX / PALM_SIZE,
      offsetY: offsetY / PALM_SIZE,
    },
  };
}

/**
 * Preprocess a cropped hand region for the Hand Landmark model.
 *
 * Steps:
 *   1. Resize to 224x224 via OffscreenCanvas
 *   2. Normalize pixels to [0, 1]: value = pixel / 255
 *
 * @param {ImageData|HTMLCanvasElement|OffscreenCanvas|ImageBitmap} source
 * @returns {Float32Array} NHWC Float32Array of shape (1, 224, 224, 3)
 */
export function preprocessLandmark(source) {
  const { canvas, ctx } = getLandmarkCanvas();

  if (source instanceof ImageData) {
    // ImageData can't be drawn directly with drawImage -- put it on a temp canvas first.
    const tmp = new OffscreenCanvas(source.width, source.height);
    const tmpCtx = tmp.getContext('2d');
    tmpCtx.putImageData(source, 0, 0);
    ctx.drawImage(tmp, 0, 0, LANDMARK_SIZE, LANDMARK_SIZE);
  } else {
    const srcW = source.videoWidth || source.width;
    const srcH = source.videoHeight || source.height;
    ctx.drawImage(source, 0, 0, srcW, srcH, 0, 0, LANDMARK_SIZE, LANDMARK_SIZE);
  }

  const imageData = ctx.getImageData(0, 0, LANDMARK_SIZE, LANDMARK_SIZE);
  const rgba = imageData.data;

  const pixelCount = LANDMARK_SIZE * LANDMARK_SIZE;
  const data = new Float32Array(pixelCount * 3);

  for (let i = 0; i < pixelCount; i++) {
    const ri = i * 4;
    const oi = i * 3;
    data[oi]     = rgba[ri]     / 255; // R
    data[oi + 1] = rgba[ri + 1] / 255; // G
    data[oi + 2] = rgba[ri + 2] / 255; // B
  }

  return data;
}
