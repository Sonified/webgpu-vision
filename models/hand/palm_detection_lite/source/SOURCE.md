# Palm Detection Lite

- **Model:** BlazePalm (palm detection, lite variant)
- **Author:** Google (MediaPipe team)
- **License:** Apache 2.0 (see [models/LICENSE](../../../LICENSE))
- **Original format:** TFLite
- **Original URL:** https://storage.googleapis.com/mediapipe-assets/palm_detection_lite.tflite
- **ONNX conversion:** Converted from TFLite using tf2onnx
- **Input:** (1, 192, 192, 3) float32, values in [0, 1]
- **Output:** 2016 anchor boxes with scores and 7 keypoints each
