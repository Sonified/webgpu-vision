# Face Detector

- **Model:** BlazeFace (short-range face detection)
- **Author:** Google (MediaPipe team)
- **License:** Apache 2.0 (see [models/LICENSE](../../../LICENSE))
- **Extracted from:** Google MediaPipe face_landmarker.task bundle
- **Bundle URL:** https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
- **ONNX conversion:** TFLite extracted from .task bundle, converted using tf2onnx
- **Input:** (1, 128, 128, 3) float32, values in [-1, 1]
- **Output:** 896 anchor boxes with scores and 6 keypoints each
