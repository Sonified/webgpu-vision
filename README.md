# WebGPU Vision

Hand and face tracking running entirely on WebGPU compute shaders in the browser. No sealed WASM binary, no WebGL, no `glReadPixels` bottleneck.

The GPU turns to the CPU and says... "Hold my bear" 🧸

## Why This Exists

MediaPipe's browser SDK uses WebGL internally for inference with synchronous readbacks. This project replaces the inference path with WebGPU compute shaders via a custom WGSL inference engine -- zero CPU readback during inference, true parallel workers, and full pipeline visibility.

## Quick Start

```bash
npm install
npm run dev
```

Open http://localhost:5173 for hand tracking, or http://localhost:5173/face.html for face tracking. Chrome 113+ (or any browser with WebGPU support). Allow camera access.

Models are included in the repo (Apache 2.0). No separate downloads needed.

## Demos

- **[`/`](index.html)**: Hand tracking wireframe overlay (hub work-in-progress, see [WORK-PLAN.md](WORK-PLAN.md))
- **[`/face.html`](face.html)**: Face landmark + blendshape wireframe overlay
- **[`/demos/ball-toss/`](demos/ball-toss/)**: Full showcase. Head-coupled 3D parallax with Three.js, hand-driven projectile throwing, MediaPipe vs WebGPU Vision A/B toggle, persisted UI settings, One Euro filtering, live `[BENCH]` lines tagged by backend and face model
- **[`/demos/hand-viz/`](demos/hand-viz/)**: Procedural energy hand visualization driven by live hand landmarks
- **[`/ar/`](ar/)**: Face puppeting demo. WebGPU Vision face landmarks drive a 52-blendshape metahuman head in Three.js

### Engine Test Pages

The [`engine/`](engine/) directory contains test harnesses and benchmarks for the custom WGSL inference engine:

- `test-hand.html`, `test-hand-sparse.html`, `test-hand-sparse-f16.html`: Hand landmark model correctness tests (standard, large, f16 variants)
- `test-face-det.html`, `test-face-lm.html`, `test-face-f16.html`: Face model correctness tests
- `bench.html`, `bench-all.html`, `test-bench.html`: Inference benchmarks
- `test-f16-shootout.html`: f32 vs f16 comparison
- `profile-dispatches.html`, `profile-face-det.html`, `profile-face.html`: GPU dispatch profiling

## Architecture

Two inference paths, both GPU-direct:

### Custom WGSL Engine (primary)

The [`engine/`](engine/) directory contains a from-scratch WGSL compute shader inference engine. Models are exported from ONNX to a JSON graph + binary weights via [`onnx_to_json.py`](engine/onnx_to_json.py), then executed entirely by hand-written WGSL shaders (conv2d, gemm, maxpool, global avg pool, fused residual blocks) dispatched by [`model-runner.js`](engine/model-runner.js). No ONNX Runtime, no WASM, no framework. Every operation is a WebGPU compute dispatch.

Supports f16 inference: a parallel set of f16 shaders (`conv2d_f16.wgsl`, `gemm_f16.wgsl`, `fused_block_f16.wgsl`, etc.) using the `shader-f16` WebGPU feature for faster inference and half the memory.

### ONNX Runtime WebGPU Path (fallback)

Uses ONNX Runtime Web with the WebGPU execution provider. Every worker shares ONNX Runtime's WebGPU device and uses `Tensor.fromGpuBuffer()` to hand compute-shader output directly to inference. Still zero CPU readbacks, but depends on the ONNX RT WASM runtime.

### Hand Tracking

Three Web Workers, all GPU-direct. Main thread is pure orchestration.

- **Palm Worker**: WebGPU compute shader letterbox -> BlazePalm inference -> anchor decode -> weighted NMS. Shared device with inference, zero readback. Fire-and-forget, never blocks tracking.
- **Landmark Worker 0 + 1**: WebGPU compute shader affine warp -> Hand Landmark inference. Both workers run in true parallel via `Promise.all` (separate worker threads).

### Face Tracking

Two Web Workers, same GPU-direct architecture.

- **Face Detection Worker**: WebGPU compute shader letterbox -> BlazeFace inference (128x128, 896 anchors) -> anchor decode -> weighted NMS. Shared device, zero readback.
- **Face Landmark Worker**: WebGPU compute shader affine warp -> Face Mesh inference (256x256, 478 landmarks). Same zero-copy path.

The face landmark model required a PReLU decomposition to run on WebGPU -- see [PRELU_DECOMPOSITION.md](PRELU_DECOMPOSITION.md).

## Pipeline

```
Camera Frame (640x480)
    |
    v
createImageBitmap (main thread, fast GPU op)
    |
    |  HAND TRACKING                         FACE TRACKING
    |                                        
    ├──> Palm Worker (fire-and-forget)       ├──> Face Detection Worker (fire-and-forget)
    |      GPU letterbox -> BlazePalm        |      GPU letterbox -> BlazeFace
    |      -> anchor decode -> weighted NMS  |      -> anchor decode -> weighted NMS
    |                                        |
    ├──> Landmark Worker 0 ──┐               └──> Face Landmark Worker
    └──> Landmark Worker 1 ──┤ Promise.all         GPU warp -> inference
                             |                     -> 478-landmark inference
    GPU warp -> inference                          -> 1434 floats return to CPU
    -> 21-landmark inference                 
    -> 63 floats return to CPU               
    |                                        
    v                                        
Main thread: landmarksToRect, draw overlay, tracking loop
```

Data never leaves GPU until the final landmark coordinates return (252 bytes per hand, 5.7KB per face).

## Performance

Benchmarked on a MacBook Pro (M1 Max, 10-core CPU, 32-core GPU, 64GB), Chrome 148, macOS 26.2, 640x480 webcam.

What matters is per-frame latency and how it maps to your actual display. On a 120Hz display, each render frame is 8.3ms. If tracking computation doesn't finish within that window, the result arrives one or more render frames late.

Live `[BENCH]` measurements from the ball-toss demo (~150 sample windows, steady state). Both backends measured on the same page, same camera feed, same hardware.

### Hand Tracking

|  | WebGPU Vision | MediaPipe |
|--|---------------|-----------|
| 1 hand | **~5ms** | ~20ms |
| 2 hands | **~8ms** | ~20ms |
| CPU readback | 252 bytes (landmarks only) | Full frame via WebGL |
| Two-hand parallel | True parallel (separate workers) | Serial (same WebGL context) |

At 120fps rendering: WebGPU Vision single-hand tracking finishes within 1 render frame (8.3ms). MediaPipe needs 3 render frames (24.9ms). That's 3x the input lag on every frame.

WebGPU Vision: WGSL engine, standard-f16 model. MediaPipe: `@mediapipe/tasks-vision` with GPU delegate (sealed binary, no precision control).

### Face Tracking (single face, 478 landmarks)

|  | WebGPU Vision | MediaPipe |
|--|---------------|-----------|
| Latency | **~13ms** | ~15ms |
| CPU readback | 5.7 KB (landmarks only) | Full frame via WebGL |

Both land within 2 render frames at 120fps (16.7ms). WebGPU Vision: WGSL engine, f16 landmarks + blendshapes. MediaPipe: same sealed SDK.

## Key Technical Decisions

- **Custom WGSL inference engine**: Hand-written compute shaders for every op (conv2d, gemm, maxpool, global avg pool, fused residual blocks) execute the model graph directly on WebGPU with no framework overhead. This is the primary inference path.
- **f16 inference**: A parallel set of f16 shaders using WebGPU's `shader-f16` feature. 50% less memory (weights and intermediates), 0.3% max coordinate error (sub-pixel), modest speed bump (~5-10% in the live pipeline). The memory savings is the real win. Default for all models.
- **Zero CPU readback, every stage**: Every worker creates its compute shaders on the same GPU device as inference. Letterbox output (detection workers) and affine-warp output (landmark workers) flow directly into the model with no intermediate CPU copy. The full cascade is GPU-resident from camera frame to final landmark coordinates.
- **Web Workers for true parallelism**: Concurrent inference calls within a single thread would deadlock. Separate workers = separate execution contexts = true parallel inference.
- **Weighted NMS** (not standard suppress and discard): overlapping detections averaged by score, matching MediaPipe's internal approach.
- **PReLU decomposition**: The face landmark model's 69 PReLU ops aren't supported by ONNX RT's WebGPU backend. Decomposing `PReLU(x, slope)` into `Relu(x) + slope * (-Relu(-x))` keeps everything on GPU. Same math, zero accuracy loss, 12x speedup (9fps to 77fps). See [PRELU_DECOMPOSITION.md](PRELU_DECOMPOSITION.md).
- **Multi-head spatial averaging (sub-pixel face position from a 128x128 model)**: BlazeFace outputs 6 keypoints plus a bbox center from independent regressor heads. Averaging all 7 into a single position estimate drops noise by sqrt(7) and gives sub-pixel-precise face tracking. Cost: 14 float adds per frame. Used in [demos/ball-toss/](demos/ball-toss/).
- **ROI tracking on the detector cascade**: After the first detection hit, subsequent frames crop around the previous bbox (3.5x margin). Stable input window = stable output, no temporal smoothing required. Falls back to full-frame search if the crop returns nothing.
- **Model URL auto-switching**: `model-urls.js` serves from local `/models/` on localhost, from `https://models.now.audio/` in production. Zero config.

## Project Structure

```
src/
  main.js                        Hand tracking entry point, webcam, render loop
  pipeline.js                    HandTracker class, orchestration, worker management
  palm-worker.js                 Palm detection worker (ONNX RT, GPU letterbox + inference)
  palm-worker-wgsl.js            Palm detection worker (WGSL engine)
  landmark-worker.js             Hand landmark worker (ONNX RT, GPU warp + inference)
  landmark-worker-wgsl.js        Hand landmark worker (WGSL engine, model tier/f16 selection)
  anchors.js                     Anchor generation + decoding for BlazePalm
  nms.js                         Weighted NMS + detectionToRect for hands
  preprocessing.js               Palm preprocessing (canvas fallback)
  face-main.js                   Face tracking entry point
  face-pipeline.js               FaceTracker class, orchestration
  face-detection-worker.js       Face detection worker (ONNX RT, GPU letterbox + BlazeFace)
  face-detection-worker-wgsl.js  Face detection worker (WGSL engine)
  face-landmark-worker.js        Face landmark worker (ONNX RT, 478-point inference)
  face-landmark-worker-wgsl.js   Face landmark worker (WGSL engine)
  face-anchors.js                Anchor generation + decoding for BlazeFace
  face-nms.js                    Weighted NMS + faceDetectionToRect
  model-urls.js                  Auto-switching model URLs (local dev / CDN production)
  settings-panel.js              In-demo settings UI (model tier, f16 toggle, etc.)
  settings-store.js              Persistent settings via localStorage
  log-gates.js                   Conditional logging (suppresses noisy logs in production)

engine/
  model-runner.js                Generic WGSL model runner (graph JSON + weights -> GPU dispatches)
  conv2d.wgsl / conv2d_f16.wgsl  Convolution compute shaders (f32 / f16)
  gemm.wgsl / gemm_f16.wgsl      Fully-connected layer shaders
  fused_block.wgsl / _f16.wgsl   Fused conv + BN + activation + residual
  maxpool.wgsl / _f16.wgsl       Max pooling
  global_avg_pool.wgsl / _f16.wgsl  Global average pooling
  add.wgsl / add_f16.wgsl        Element-wise add
  onnx_to_json.py                ONNX model -> JSON graph + binary weights converter
  palm-detector.js               Standalone palm detection via WGSL engine
  inference.js                   Low-level inference utilities

demos/
  ball-toss/                     Showcase: 3D parallax + hand projectiles + A/B comparison
  hand-viz/                      Procedural energy hand visualization

ar/                              Face puppeting: blendshape-driven metahuman head

models/                          ONNX model files (Apache 2.0, see models/LICENSE)
benchmark/                       MediaPipe comparison baselines
diagnostic/                      Debugging tools (ONNX dumps, WGSL layer dumps, captures)
```

## Models

All models are Apache 2.0 licensed. Hand models from [OpenCV Zoo](https://github.com/opencv/opencv_zoo). Face models converted from Google's [MediaPipe .task bundle](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task) via tf2onnx, with PReLU decomposition for WebGPU compatibility.

### Hand Models

| Model | Input | Params | Variants | Size (f32 / f16) |
|-------|-------|--------|----------|-------------------|
| Palm detection | (1, 192, 192, 3) float32 | -- | f32 only | 3.7 MB |
| Hand landmark standard | (1, 224, 224, 3) float32 | 1,011,716 | f32, f16 | 3.9 MB / 1.9 MB |
| Hand landmark large | (1, 224, 224, 3) float32 | 2,643,045 | f32, f16 | 10 MB / 5 MB |

**Standard** is Google's "full" hand landmark model. **Large** is their bigger "sparse" variant with 2.6x more parameters and significantly better pinch/thumb tracking, at the cost of ~3ms extra inference time.

The in-demo settings panel lets you switch between `standard-f32`, `standard-f16`, `large-f32`, and `large-f16` at runtime.

### Face Models

| Model | Input | Variants | Size |
|-------|-------|----------|------|
| Face detection | (1, 128, 128, 3) float32 | f32 only | 409 KB |
| Face landmarks | (1, 256, 256, 3) float32 | f32, f16 | 4.8 MB / 2.4 MB |

## Requirements

- Chrome 113+, Edge 113+, or Safari 18+ (WebGPU support)
- A device with a camera
- Node.js 18+ (for the dev server)

## Acknowledgments

- Google MediaPipe team for the trained models and published research (Apache 2.0)
- [OpenCV Zoo](https://github.com/opencv/opencv_zoo) for hand model ONNX conversions
- [PINTO0309](https://github.com/PINTO0309) for the reference Python implementation
- [geaxgx](https://github.com/geaxgx/depthai_hand_tracker) for the clearest reference glue code
- Microsoft ONNX Runtime team for the WebGPU execution provider

## License

MIT
