# WebGPU Hand Tracking

Real-time hand tracking in the browser via WebGPU compute shaders. No WebGL, no WASM black box, no `glReadPixels` bottleneck.

Runs Google's BlazePalm + Hand Landmark neural networks through ONNX Runtime Web with the WebGPU backend. The full MediaPipe hand tracking pipeline, rebuilt from scratch for GPU-native web.

## Quick Start

```bash
# Install dependencies
npm install

# Download the ONNX models into models/
curl -L "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/palm_detection_mediapipe/palm_detection_mediapipe_2023feb.onnx" \
  -o models/palm_detection_lite.onnx

curl -L "https://huggingface.co/opencv/handpose_estimation_mediapipe/resolve/main/handpose_estimation_mediapipe_2023feb.onnx" \
  -o models/hand_landmark_full.onnx

# Start the dev server
npm run dev
```

Open http://localhost:5173 in Chrome 113+ (or any browser with WebGPU support). Allow camera access. You should see your webcam feed with hand tracking overlays.

## How It Works

Two neural networks in a cascade:

1. **BlazePalm** (palm detection) -- finds where hands are in the frame. 192x192 input, 2016 anchor boxes, weighted NMS.
2. **Hand Landmark** -- given a cropped hand region, extracts 21 3D keypoints. 224x224 input, runs per detected hand.

The tracking loop makes it fast: once a hand is found, the landmark model's output defines the crop region for the next frame. Palm detection only re-runs when a hand is lost or every 30 frames as a safety net.

```
Camera Frame (640x480)
    |
    v
Letterbox pad to square, resize to 192x192
    |
    v
BlazePalm (WebGPU inference)
    |
    v
Anchor decode + Weighted NMS + Rotation
    |
    v
Affine warp (crop rotated hand region to 224x224)
    |
    v
Hand Landmark (WebGPU inference)
    |
    v
21 3D keypoints, projected back to video space
    |
    v
Next frame: use landmarks as crop region (skip palm detection)
```

## Why This Exists

MediaPipe's browser SDK uses WebGL internally for inference. Even with "GPU delegate," it does synchronous `glReadPixels` readbacks that cost 8-22ms per call. Two-hand tracking drops to ~15fps. The WASM binary is sealed -- you can't optimize what you can't see.

This project replaces that entire inference path with WebGPU compute shaders via ONNX Runtime Web. No synchronous readbacks. Full pipeline visibility. The tracking loop keeps landmark-only frames fast by skipping palm detection.

## Project Structure

```
src/
  main.js           Entry point, webcam setup, render loop
  pipeline.js        HandTracker class, model loading, inference orchestration
  anchors.js         Anchor generation + detection decoding for BlazePalm
  nms.js             Weighted NMS + detection-to-rotated-rect conversion
  preprocessing.js   Frame preprocessing (letterbox, resize, normalize)
models/              ONNX model files (not checked in, see Quick Start)
```

## Requirements

- Chrome 113+, Edge 113+, or Safari 18+ (WebGPU support)
- A device with a camera
- Node.js 18+ (for the dev server)

## Models

Uses Google's MediaPipe hand tracking models converted to ONNX format by the OpenCV Zoo team (Apache 2.0 license):

| Model | Input | Output | Size |
|-------|-------|--------|------|
| Palm detection | (1, 192, 192, 3) float32 [0,1] | 2016 anchor boxes + scores | 3.7 MB |
| Hand landmark | (1, 224, 224, 3) float32 [0,1] | 21 keypoints (x,y,z) + hand flag + handedness | 3.9 MB |

## Acknowledgments

- Google MediaPipe team for the trained models and published research
- [OpenCV Zoo](https://github.com/opencv/opencv_zoo) for ONNX model conversions
- [PINTO0309](https://github.com/PINTO0309) for the reference Python implementation
- [geaxgx](https://github.com/geaxgx/depthai_hand_tracker) for the clearest reference glue code
- Microsoft ONNX Runtime team for the WebGPU execution provider

## License

MIT
