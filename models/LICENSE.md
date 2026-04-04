# Model Licenses

## Hand Tracking Models

- `palm_detection_lite.onnx` -- Palm detection model
- `hand_landmark_full.onnx` -- Hand landmark model

Source: [OpenCV Zoo](https://github.com/opencv/opencv_zoo) (ONNX conversions of Google MediaPipe models)
License: Apache 2.0

## Face Tracking Models

- `face_detector.onnx` -- Face detection model (BlazeFace short-range)
- `face_landmarks_detector.onnx` -- Face landmark model (478 landmarks, PReLU decomposed)

Source: Converted from Google MediaPipe's [face_landmarker.task](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task) bundle using tf2onnx. PReLU ops decomposed into Relu+Neg+Mul+Add for WebGPU compatibility. See [PRELU_DECOMPOSITION.md](../PRELU_DECOMPOSITION.md) for details.
License: Apache 2.0 (Google MediaPipe)
