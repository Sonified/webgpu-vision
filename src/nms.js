/**
 * Weighted NMS and detection-to-rotated-rect conversion.
 * Ported from MediaPipe's pipeline via ARCHITECTURE.md reference code.
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
 * Convert a palm detection to a rotated rectangle for the landmark model.
 * Ported from geaxgx/depthai_hand_tracker mediapipe_utils.py:
 *   detections_to_rect() + rect_transformation()
 *
 * Detection coords must already be in video-normalized space [0,1].
 *
 * @param {Object} detection - {cx, cy, w, h, score, keypoints}
 * @param {number} imgW - video width in pixels
 * @param {number} imgH - video height in pixels
 * @returns {Object} {cx, cy, w, h, angle} in pixel coordinates
 */
export function detectionToRect(detection, imgW, imgH) {
  const wrist = detection.keypoints[0];
  const middle = detection.keypoints[2];

  // Rotation: target_angle(90deg) - atan2(-(y1-y0), x1-x0)
  const rotation = Math.PI / 2 - Math.atan2(-(middle.y - wrist.y), middle.x - wrist.x);
  // Normalize to [-pi, pi]
  const angle = rotation - 2 * Math.PI * Math.floor((rotation + Math.PI) / (2 * Math.PI));

  // Rect center from detection box (in normalized coords)
  let rcx = detection.cx;
  let rcy = detection.cy;
  const rw = detection.w;
  const rh = detection.h;

  // Shift center by -0.5 * height in rotated direction (in pixel space)
  const shiftX = 0;
  const shiftY = -0.5;
  const xShift = imgW * rw * shiftX * Math.cos(angle) - imgH * rh * shiftY * Math.sin(angle);
  const yShift = imgW * rw * shiftX * Math.sin(angle) + imgH * rh * shiftY * Math.cos(angle);
  const cxPx = rcx * imgW + xShift;
  const cyPx = rcy * imgH + yShift;

  // square_long: use the longer side, then scale by 2.9
  const longSide = Math.max(rw * imgW, rh * imgH);
  const size = longSide * 2.9;

  return { cx: cxPx, cy: cyPx, w: size, h: size, angle };
}
