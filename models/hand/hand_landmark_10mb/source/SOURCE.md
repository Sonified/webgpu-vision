# Hand Landmark Sparse (10MB)

- **Model:** MediaPipe Hand Landmark (sparse variant, 2.6M params)
- **Author:** Google (MediaPipe team)
- **License:** Apache 2.0 (see [models/LICENSE](../../../LICENSE))
- **ONNX conversion by:** PINTO0309 (Katsuya Hyodo)
- **Downloaded from:** https://github.com/PINTO0309/hand_landmark/releases/download/1.0.0/hand_landmark_sparse_Nx3x224x224.onnx
- **Converter repo:** https://github.com/PINTO0309/hand_landmark
- **Original format:** TFLite (Google), converted to ONNX by PINTO0309 using tflite2tensorflow/tf2onnx
- **Input:** (1, 3, 224, 224) float32, values in [0, 1], NCHW layout
- **Output:** 21 3D hand landmarks, hand presence flag, handedness, world landmarks
- **Notes:** Vastly better pinch and thumb tracking than the 4MB full variant. Not yet ported to WGSL engine (tested via ORT only).
