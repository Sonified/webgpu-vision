# Face Blendshapes

- **Model:** MediaPipe Face Blendshapes (52 expression coefficients)
- **Author:** Google (MediaPipe team)
- **License:** Apache 2.0 (see [models/LICENSE](../../../LICENSE))
- **Extracted from:** Google MediaPipe face_landmarker.task bundle
- **Bundle URL:** https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
- **ONNX conversion:** TFLite extracted from .task bundle, converted using tf2onnx
- **Input:** 146 face landmarks (subset selected by the face landmark model)
- **Output:** 52 blendshape coefficients (facial expressions)
