// Model URL resolution: local in dev, CDN in production.

const CDN_BASE = 'https://models.now.audio';
const LOCAL_BASE = '/models';

const isLocal = typeof location !== 'undefined' && location.hostname === 'localhost';
const BASE = isLocal ? LOCAL_BASE : CDN_BASE;

export const PALM_MODEL_URL = `${BASE}/hand/palm_detection_lite/source/palm_detection_lite.onnx`;
export const HAND_LANDMARK_URL = `${BASE}/hand/hand_landmark_4mb/source/hand_landmark_full.onnx`;
export const FACE_DETECTOR_URL = `${BASE}/face/face_detector/source/face_detector.onnx`;
export const FACE_LANDMARK_URL = `${BASE}/face/face_landmarks/source/face_landmarks_detector.onnx`;
export const FACE_BLENDSHAPE_URL = `${BASE}/face/face_blendshapes/source/face_blendshapes.onnx`;
