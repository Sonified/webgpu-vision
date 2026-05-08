# Face Landmarks Detector

- **Model:** FaceMesh V2 (478 3D face landmarks)
- **Author:** Google (MediaPipe team)
- **License:** Apache 2.0 (see [models/LICENSE](../../../LICENSE))
- **Extracted from:** Google MediaPipe face_landmarker.task bundle
- **Bundle URL:** https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
- **ONNX conversion:** TFLite extracted from .task bundle, converted using tf2onnx
- **Modification:** 69 PReLU ops decomposed into Relu+Neg+Mul+Add (mathematically identical, zero accuracy loss) for WebGPU compatibility. See PRELU_DECOMPOSITION.md in the repo root.
- **Input:** (1, 256, 256, 3) float32, values in [0, 1]
- **Output:** 478 3D face landmarks, face presence flag, blendshape input landmarks
