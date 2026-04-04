/**
 * Weighted NMS and detection-to-rotated-rect conversion for face detection.
 * Adapted from nms.js for face pipeline.
 */

/**
 * Compute intersection-over-union between two center-format boxes.
 */
function computeIoU(a, b) {
  const ax1 = a.cx - a.w / 2, ay1 = a.cy - a.h / 2;
  const ax2 = a.cx + a.w / 2, ay2 = a.cy + a.h / 2;
  const bx1 = b.cx - b.w / 2, by1 = b.cy - b.h / 2;
  const bx2 = b.cx + b.w / 2, by2 = b.cy + b.h / 2;

  const ix1 = Math.max(ax1, bx1), iy1 = Math.max(ay1, by1);
  const ix2 = Math.min(ax2, bx2), iy2 = Math.min(ay2, by2);
  const interW = Math.max(0, ix2 - ix1);
  const interH = Math.max(0, iy2 - iy1);
  const inter = interW * interH;

  const areaA = a.w * a.h;
  const areaB = b.w * b.h;
  const union = areaA + areaB - inter;

  return union > 0 ? inter / union : 0;
}

/**
 * Weighted NMS -- overlapping detections are averaged by score,
 * not simply suppressed. This is what MediaPipe uses internally.
 *
 * @param {Array} detections - Array of {cx, cy, w, h, score, keypoints}
 * @param {number} iouThreshold - IoU threshold for clustering (default 0.3)
 * @returns {Array} Filtered detections with weighted-average boxes
 */
export function weightedNMS(detections, iouThreshold = 0.3) {
  detections = detections.slice().sort((a, b) => b.score - a.score);
  const kept = [];

  while (detections.length > 0) {
    const best = detections.shift();
    const cluster = [best];

    detections = detections.filter(d => {
      if (computeIoU(best, d) > iouThreshold) {
        cluster.push(d);
        return false;
      }
      return true;
    });

    // Weighted average of cluster
    let totalW = 0;
    let cx = 0, cy = 0, w = 0, h = 0;
    for (const d of cluster) {
      cx += d.cx * d.score;
      cy += d.cy * d.score;
      w += d.w * d.score;
      h += d.h * d.score;
      totalW += d.score;
    }
    kept.push({
      cx: cx / totalW, cy: cy / totalW,
      w: w / totalW, h: h / totalW,
      score: best.score,
      keypoints: best.keypoints, // use highest-score keypoints
    });
  }
  return kept;
}

/**
 * Convert a face detection to a rotated rectangle for the landmark model.
 * Uses left eye (keypoint 0) and right eye (keypoint 1) for rotation.
 *
 * Detection coords must already be in video-normalized space [0,1].
 *
 * @param {Object} detection - {cx, cy, w, h, score, keypoints}
 * @param {number} imgW - video width in pixels
 * @param {number} imgH - video height in pixels
 * @returns {Object} {cx, cy, w, h, angle} in pixel coordinates
 */
export function faceDetectionToRect(detection, imgW, imgH) {
  // Keypoint 0 = right eye, 1 = left eye (from camera's perspective)
  const rightEye = detection.keypoints[0];
  const leftEye = detection.keypoints[1];
  const angle = Math.atan2(leftEye.y - rightEye.y, leftEye.x - rightEye.x);
  const longSide = Math.max(detection.w * imgW, detection.h * imgH);
  const size = longSide * 1.5;
  const cx = detection.cx * imgW;
  const cy = detection.cy * imgH;
  return { cx, cy, w: size, h: size, angle };
}
