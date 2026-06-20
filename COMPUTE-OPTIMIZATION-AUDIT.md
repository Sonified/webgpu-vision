# Compute Optimization Audit

**Date**: 2026-06-20
**Baseline tag**: `pre-wasm-optimization` (commit `b4eeb35`)
**Scope**: Full pass through the WebGPU Vision library API -- every file in `src/` and `engine/` -- looking for JavaScript compute that should live on GPU (WGSL), in Rust/WASM, or in a hybrid pipeline.

---

## Architecture Context

The library uses a clean worker-based architecture with three tiers:

```
Main thread (pipeline.js)         -- pure orchestration, slot tracking, identity
  |
  +-- Palm detection worker       -- GPU letterbox -> inference -> JS decode -> JS NMS
  |     palm-worker.js (ORT)
  |     palm-worker-wgsl.js (native WGSL engine)
  |
  +-- Landmark workers (x2)       -- GPU affine warp -> inference -> JS projection
  |     landmark-worker.js (ORT)
  |     landmark-worker-wgsl.js (native WGSL engine)
  |
  +-- Face detection worker       -- same pattern as palm
  +-- Face landmark workers       -- same pattern as hand landmark
  +-- Blendshape worker           -- fire-and-forget async inference
```

**What's already GPU-optimized** (and should not be touched):
- All neural network inference (WGSL engine with pre-compiled dispatch replay)
- Camera frame preprocessing (letterbox/affine warp via WGSL compute shaders)
- Zero-copy tensor handoff (warp output buffer IS the inference input buffer)
- Texture and bind group caching in WGSL workers
- Operator fusion in model-runner.js (Conv+PReLU, Conv+ReLU6+Add, fused residual blocks)

**What's still on the CPU** (and is the subject of this audit):
- Post-inference decode (anchor decode, sigmoid, score filter)
- Non-maximum suppression
- Detection-to-rect conversion
- Affine parameter computation
- Landmark projection (inverse affine)
- Result packaging (Float32Array -> {x,y,z} objects)
- Preview image format conversion

---

## Findings

Sorted by impact-to-effort ratio. Quick wins first.

---

### Finding 1: Preview Readback Runs Unconditionally

**Priority**: Do first (trivial, immediate win)

| | |
|---|---|
| **File** | `src/palm-worker-wgsl.js` lines 206-226 |
| **What it does now** | Every palm detection frame, copies the entire letterbox buffer (192x192x3 floats = 442KB) to a staging buffer via `copyBufferToBuffer`, then reads it back with `mapAsync`, then converts NCHW float32 to RGBA uint8 in a 36,864-iteration JS loop. This happens even when nobody looks at the preview. |
| **Problem** | ~0.5ms GPU copy + ~0.5ms CPU conversion, every detection frame, for a debug feature |
| **Fix** | Add a `wantPreview` flag to the `detect` message. Skip the `copyBufferToBuffer` call and the `mapAsync`/conversion block when false. |
| **Expected impact** | Saves ~1ms per palm detection frame. More importantly, eliminates a `mapAsync` that can stall the GPU pipeline. |
| **Migration complexity** | Trivial. Boolean flag, conditional block. |

**Current code** (the hot path that should be conditional):
```js
// line 206: this copy should only happen when preview is requested
enc.copyBufferToBuffer(letterboxOutputBuf, 0, previewReadBuf, 0, previewReadBuf.size);

// lines 212-226: this entire block should be gated
await previewReadBuf.mapAsync(GPUMapMode.READ);
const nchw = new Float32Array(previewReadBuf.getMappedRange().slice(0));
previewReadBuf.unmap();
const S = PALM_SIZE * PALM_SIZE;
const rgba = new Uint8ClampedArray(S * 4);
for (let i = 0; i < S; i++) {
  rgba[i * 4]     = nchw[i] * 255;
  rgba[i * 4 + 1] = nchw[S + i] * 255;
  rgba[i * 4 + 2] = nchw[2 * S + i] * 255;
  rgba[i * 4 + 3] = 255;
}
```

---

### Finding 2: Per-Frame Object Allocation for Landmarks

**Priority**: Do second (low effort, meaningful GC impact)

| | |
|---|---|
| **Files** | `src/pipeline.js` lines 148-170 (hands), `src/face-pipeline.js` lines 123-131 (faces) |
| **What it does now** | Receives a flat `Float32Array` buffer from the worker via `postMessage` transfer. Immediately iterates and creates `{x, y, z}` objects for every landmark. Hands: 21 objects per hand, up to 42/frame. Faces: 478 objects per face. At 60fps with one hand and one face: **(42 + 478) x 60 = 31,200 short-lived object allocations per second**. |
| **Problem** | GC nursery pressure. These objects are born and die within a single frame. The minor GC pauses show up as occasional dropped frames (16ms -> 20ms+). |
| **Fix** | Keep landmarks as flat `Float32Array` throughout the pipeline. Expose a view accessor if consumers need `landmarks[i].x` syntax. Two options: |

**Option A** -- Lazy accessor (backward compatible):
```js
class LandmarkView {
  constructor(buffer, count) {
    this._buf = buffer; // Float32Array, length = count * 3
    this.length = count;
  }
  get(i) { // or use a Proxy for [i] access
    const o = i * 3;
    return { x: this._buf[o], y: this._buf[o+1], z: this._buf[o+2] };
  }
}
```

**Option B** -- Flat API (breaking change, better for consumers):
```js
// result.landmarks is Float32Array(63) for 21 landmarks
// Access: landmarks[i*3] = x, landmarks[i*3+1] = y, landmarks[i*3+2] = z
```

| **Expected impact** | Eliminates ~31K object allocations per second. Removes a class of GC pauses that cause frame drops. |
| **Migration complexity** | Low. The workers already send flat arrays. The objectification happens in `_onMessage` handlers. The change is removing code, not adding it. |

**Where the objects are created** (pipeline.js):
```js
// lines 148-158 -- this loop creates 21 objects per hand per frame
if (e.data.landmarks) {
  const flat = new Float32Array(e.data.landmarks);
  for (let i = 0; i < 21; i++) {
    landmarks.push({
      x: flat[i * 3],
      y: flat[i * 3 + 1],
      z: flat[i * 3 + 2],
    });
  }
}
```

**Where they're created for faces** (face-pipeline.js):
```js
// lines 123-131 -- 478 objects per face per frame
if (e.data.landmarks) {
  const flat = new Float32Array(e.data.landmarks);
  for (let i = 0; i < 478; i++) {
    landmarks.push({
      x: flat[i * 3],
      y: flat[i * 3 + 1],
      z: flat[i * 3 + 2],
    });
  }
}
```

---

### Finding 3: Anchor Decode + Score Filter on GPU

**Priority**: Highest-impact single change

| | |
|---|---|
| **Files** | `src/anchors.js` lines 58-86 (called from both palm workers) |
| **What it does now** | After palm detection inference, reads back ALL 2016 anchor outputs from GPU to CPU: regressors (2016 x 18 = 36,288 floats) + scores (2016 x 1). Then iterates all 2016 anchors in JS: applies sigmoid to each score, filters by 0.5 threshold, decodes bounding box + 7 keypoints for survivors. Typically only 0-5 detections survive. |
| **Problem** | Reading back **153KB** (38,304 floats x 4 bytes) when only ~200 bytes of that data matters. The `mapAsync` for this readback is the single largest CPU-GPU sync point in the palm detection path. Plus ~2016 sigmoid calculations in JS. |
| **Fix** | WGSL compute shader with three stages: |

```
Stage 1: Decode + Filter (one thread per anchor, 2016 threads)
  - sigmoid(score)
  - if score < threshold: discard
  - decode box: raw / 192.0 + anchor_center
  - decode 7 keypoints: same transform
  - atomicAdd to a counter, write to compacted output

Stage 2: Read back only the counter + surviving detections
  - Counter is 1 u32
  - Each detection is ~19 floats (cx, cy, w, h, score, 7 keypoints x 2)
  - Typical readback: 4 + (5 x 19 x 4) = 384 bytes vs 153KB

Stage 3: NMS stays on CPU
  - Sequential algorithm, operates on <10 items
  - Not worth GPU-ifying
```

| **Expected impact** | Reduces palm detection readback from **153KB to ~400 bytes** (375x reduction). Eliminates the JS decode loop. Removes the longest `mapAsync` stall in the detection path. |
| **Migration complexity** | Medium. Needs: (a) a new WGSL shader for decode+filter with atomic compaction, (b) two small GPU buffers (counter + compacted output), (c) modified readback in palm workers. The anchors are static and can be baked into a GPU buffer at init time. |

**Current JS code that moves to GPU** (anchors.js):
```js
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

export function decodeDetections(regressors, scores, anchors) {
  const detections = [];
  for (let i = 0; i < 2016; i++) {
    const score = sigmoid(scores[i]);
    if (score < 0.5) continue;  // <-- 99%+ of anchors filtered here

    const ax = anchors[i * 2];
    const ay = anchors[i * 2 + 1];
    const ri = i * VALUES_PER_ANCHOR;
    const cx = regressors[ri + 0] / 192 + ax;
    const cy = regressors[ri + 1] / 192 + ay;
    const w  = regressors[ri + 2] / 192;
    const h  = regressors[ri + 3] / 192;
    // ... 7 keypoints decoded similarly
  }
  return detections;
}
```

**Proposed WGSL shader sketch**:
```wgsl
@group(0) @binding(0) var<storage, read>       regressors: array<f32>;  // 2016 x 18
@group(0) @binding(1) var<storage, read>       scores:     array<f32>;  // 2016
@group(0) @binding(2) var<storage, read>       anchors:    array<f32>;  // 2016 x 2
@group(0) @binding(3) var<storage, read_write> output:     array<f32>;  // compacted detections
@group(0) @binding(4) var<storage, read_write> counter:    atomic<u32>; // number of survivors

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= 2016u) { return; }

  let raw_score = scores[i];
  let score = 1.0 / (1.0 + exp(-raw_score));
  if (score < 0.5) { return; }

  let ax = anchors[i * 2u];
  let ay = anchors[i * 2u + 1u];
  let ri = i * 18u;

  let slot = atomicAdd(&counter, 1u);
  let oi = slot * 19u;  // 19 floats per detection
  output[oi]      = regressors[ri] / 192.0 + ax;       // cx
  output[oi + 1u] = regressors[ri + 1u] / 192.0 + ay;  // cy
  output[oi + 2u] = regressors[ri + 2u] / 192.0;       // w
  output[oi + 3u] = regressors[ri + 3u] / 192.0;       // h
  output[oi + 4u] = score;
  // ... 7 keypoints
}
```

---

### Finding 4: Texture + BindGroup Allocation Per Frame (ORT Path)

**Priority**: Low -- ORT workers are baseline infrastructure, not the optimization target. The WGSL workers already cache correctly. The ORT palm worker is still selectable at runtime (engine dropdown), so these aren't dead code, but optimizing them is unlikely to be worth the effort. Noted here for completeness.

| | |
|---|---|
| **Files** | `src/landmark-worker.js` lines 125-162, `src/palm-worker.js` lines 112-136 |
| **What it does now** | Creates a new `GPUTexture` + `GPUBindGroup` every inference call in the ORT-backed workers. Destroys the texture after use. |
| **Problem** | GPU-side allocation overhead per frame. Texture creation involves driver calls. |
| **Fix** | Cache the texture and bind group, recreate only when video resolution changes. This is exactly what `landmark-worker-wgsl.js` lines 241-273 already does with `cachedWarpTexture` / `cachedWarpBindGroup` / `cachedWarpSize`. Copy that pattern. |
| **Expected impact** | Low (ORT paths are not the default), but eliminates unnecessary driver calls. |
| **Migration complexity** | Trivial. The pattern is already implemented in the WGSL workers. |

---

### Finding 5: Per-Frame Float32Array Allocations for Uniforms (ORT Path)

**Priority**: Low -- same rationale as Finding 4. ORT path just needs to work. Unlikely to implement.

| | |
|---|---|
| **Files** | `src/landmark-worker.js` lines 137-140, `src/palm-worker.js` lines 124-126 |
| **What it does now** | `new Float32Array([inv.a, inv.b, inv.c, 0, ...])` every frame to write uniforms. |
| **Fix** | Pre-allocate a scratch array at module scope (as `landmark-worker-wgsl.js` line 245 does with `const warpUniforms = new Float32Array(12)`). Write into it, then pass to `writeBuffer`. |
| **Expected impact** | Eliminates one 48-byte allocation per frame per worker. Prevents GC nursery churn. |
| **Migration complexity** | Trivial. |

**landmark-worker.js currently**:
```js
gpuDevice.queue.writeBuffer(uniformBuffer, 0, new Float32Array([
  inv.a, inv.b, inv.c, 0,
  inv.d, inv.e, inv.f, 0,
  bitmap.width, bitmap.height, 0, 0,
]));
```

**Should be**:
```js
const warpUniforms = new Float32Array(12); // module scope, allocated once

// in dispatchWarp():
warpUniforms[0] = inv.a; warpUniforms[1] = inv.b; warpUniforms[2] = inv.c;
warpUniforms[4] = inv.d; warpUniforms[5] = inv.e; warpUniforms[6] = inv.f;
warpUniforms[8] = bitmap.width; warpUniforms[9] = bitmap.height;
gpuDevice.queue.writeBuffer(uniformBuffer, 0, warpUniforms);
```

---

### Finding 6: NCHW->RGBA Preview Conversion on GPU

**Priority**: Low (only matters if preview stays in the hot path)

| | |
|---|---|
| **File** | `src/palm-worker-wgsl.js` lines 218-226 |
| **What it does now** | After reading back the NCHW letterbox data, runs a 36,864-iteration JS loop to convert to RGBA uint8. |
| **Fix** | If the preview feature needs to stay: write a WGSL compute shader or render pass that samples from the NCHW buffer and writes to an `rgba8unorm` storage texture. The GPU does the channel swizzle + float-to-uint8 conversion in one dispatch. Read back the RGBA directly. |
| **Expected impact** | Low. This only fires on palm detection frames (not every render frame), and Finding 1 should gate it behind a flag anyway. |
| **Migration complexity** | Low. Standard compute shader pattern. |

---

### Finding 7: Rust -> WASM for Post-Processing Chain

**Priority**: Future work (high effort, moderate payoff)

| | |
|---|---|
| **Files** | `src/anchors.js`, `src/nms.js`, `src/face-nms.js`, affine math in landmark workers |
| **What it does now** | The entire CPU-side post-processing chain runs in vanilla JS: anchor decode -> sigmoid -> filter -> weighted NMS -> detectionToRect -> computeAffineParams -> landmark projection. Each piece is small, but they chain together on the hot path. |
| **Where it should live** | A single Rust WASM module compiled with `wasm-pack`. One function takes raw GPU outputs (as `&[f32]` slices), runs the full chain, returns projected landmarks + rects. Benefits: (a) no per-element number boxing, (b) SIMD via `std::simd` or `packed_simd`, (c) zero-alloc with pre-allocated output buffers, (d) the sigmoid loop is perfectly SIMD-able (4-wide or 8-wide). |
| **Expected impact** | Medium. The JS chain totals maybe 1-2ms. WASM with SIMD would do it in ~0.3-0.5ms. Savings: ~1ms per detection frame. More impactful on lower-end devices where JS JIT is weaker. |
| **Migration complexity** | High. Adds a Rust + wasm-pack build toolchain. Needs careful interface design for passing typed arrays across the JS/WASM boundary without copying. Should not be attempted until Findings 1-3 are done -- those eliminate most of the work this would optimize. |

**What the Rust module would look like**:
```rust
#[wasm_bindgen]
pub fn decode_and_project(
    regressors: &[f32],    // 2016 x 18
    scores: &[f32],        // 2016
    anchors: &[f32],       // 2016 x 2 (static, passed once)
    landmarks_raw: &[f32], // 63 floats from landmark model
    inv_affine: &[f32],    // 6 floats (a, b, c, d, e, f)
    vw: f32, vh: f32,
    out_detections: &mut [f32],  // pre-allocated
    out_landmarks: &mut [f32],   // pre-allocated, 63 floats
) -> u32 { /* returns detection count */ }
```

**Note**: If Finding 3 (GPU anchor decode) is implemented, the WASM module shrinks to just the affine math + NMS, which makes the build toolchain harder to justify. Evaluate after Finding 3 ships.

---

### Finding 8: Pipeline Parallelism (CPU/GPU Overlap)

**Priority**: Future architecture (high effort, high payoff at scale)

| | |
|---|---|
| **File** | `src/pipeline.js` lines 249-540 |
| **What it does now** | `processFrame()` is strictly sequential within a frame: create VideoFrame -> post to worker -> await inference result -> compute next-frame ROI -> return. The main thread and CPU cores are idle during the ~5-10ms GPU inference. |
| **What it could do** | While GPU processes frame N, the CPU prepares frame N+1: (a) compute affine params from frame N-1's landmarks (already known), (b) create the VideoFrame, (c) post the warp uniforms to the worker so the warp dispatch can start immediately when inference N completes. This is the classic "double-buffer" or "pipeline-ahead" pattern from game engines. |
| **Expected impact** | Reduces perceived latency by 1-2ms per frame by hiding CPU prep behind GPU execution. Most valuable at high frame rates (90fps+) or on devices where CPU and GPU can truly run in parallel (discrete GPU setups, though rare in browser contexts). |
| **Migration complexity** | High. Requires restructuring `processFrame` from a request/response model to a streaming pipeline. The slot assignment logic (which depends on this frame's results) makes prefetching tricky -- you'd need speculative ROIs that can be corrected. This is "Phase 5" level work and should wait until the simpler wins are shipped. |

---

## What Does NOT Need Optimization

These areas were audited and found to be already well-optimized or too small to matter:

| Area | Why it's fine |
|---|---|
| **Weighted NMS** (`nms.js`, `face-nms.js`) | Sequential algorithm, operates on <10 items post-filter. O(n^2) on n<10 is ~100 operations. Not worth GPU-ifying. |
| **Landmark projection** (landmark workers, 21 x affine transform) | 63 multiply-adds. Sub-microsecond. The data is already on CPU for postMessage. |
| **landmarksToRect()** (`pipeline.js` lines 543-590) | ~50 operations, called 1-2x per frame. Pure geometry. |
| **computeAffineParams()** (landmark workers) | ~30 operations per call. Matrix inversion of a 2x3. |
| **Slot assignment / centroid tracking** (`pipeline.js`) | Max 2 slots, max ~5 detections. O(n*m) where n,m < 5. |
| **generateAnchors()** (`anchors.js`) | Called once at init. 2016 anchors computed in <1ms. Not in hot path. |
| **ModelRunner compiled replay** (`engine/model-runner.js` `encodeInto`) | Already optimal: flat loop over pre-built steps, no allocation, no branching. |
| **Uniform buffer pool** (`engine/model-runner.js` `_getUniformBuf`) | Already pooled and reused across frames. |
| **Texture caching in WGSL workers** | Already implemented correctly in both `landmark-worker-wgsl.js` and `palm-worker-wgsl.js`. |

---

## Recommended Execution Order

```
Session 1 (quick wins, <1 hour):
  [1] Gate preview readback behind flag

Session 2 (GC reduction, ~2 hours):
  [2] Flat landmark arrays through pipeline
      - Decide on API surface (LandmarkView vs raw Float32Array)
      - Update pipeline.js, face-pipeline.js
      - Update demo consumers (main.js, ball-toss)

Session 3 (GPU decode, ~4 hours):
  [3] WGSL anchor decode + score filter shader
      - Write decode_filter.wgsl
      - Add anchor GPU buffer at init
      - Add counter + compacted output buffers
      - Modify palm-worker-wgsl.js readback
      - Verify against current JS decode output

Future:
  [6] GPU preview conversion (if preview stays)
  [7] WASM post-processing (evaluate after [3] ships)
  [8] Pipeline parallelism (Phase 5 architecture)

Deprioritized (ORT path -- baseline only, unlikely to implement):
  [4] Cache texture/bindgroup in ORT workers
  [5] Pre-allocate uniform scratch arrays in ORT workers
```

---

## Metrics to Track

Before and after each change, measure:

1. **Palm detection round-trip** (ms): `performance.now()` around the full detect cycle in palm worker
2. **Landmark round-trip** (ms): same for landmark inference
3. **GC pause frequency**: Chrome DevTools Performance tab, look for minor GC events per second
4. **Readback bytes per frame**: total bytes transferred GPU -> CPU via `mapAsync`
5. **Steady-state FPS**: 10-second average with two hands tracked

### Baseline Numbers

Captured 2026-06-20, tag `pre-wasm-optimization`, **battery power** (unplugged), MacBook.
Config: hand tracking (standard-f32, 1.5-2 hands) + face tracking (landmark f16) + blendshapes.

**Hand tracking** (`WGPU Hand[standard-f32]`):
| Window | Hands | Mean | P95 | Max | Samples |
|--------|-------|------|------|------|---------|
| Warmup | 1.5h | 6.7ms | 12.3ms | 14.3ms | 109 |
| Steady 1 | 2h | 7.8ms | 12.3ms | 17.7ms | 150 |
| Steady 2 | 2h | 8.0ms | 12.6ms | 17.9ms | 150 |
| Steady 3 | 2h | 8.1ms | 12.8ms | 14.2ms | 151 |
| Steady 4 | 2h | 8.2ms | 12.4ms | 13.6ms | 149 |
| Late | 1.5h | 6.0ms | 10.8ms | 14.2ms | 151 |

**Face tracking** (`WGPU Face[LAN:f16]`):
| Window | Mean | P95 | Max | Samples |
|--------|------|------|------|---------|
| Warmup | 11.9ms | 14.8ms | 25.8ms | 109 |
| Steady 1 | 13.1ms | 15.9ms | 24.9ms | 150 |
| Steady 2 | 13.1ms | 16.6ms | 23.6ms | 150 |
| Steady 3 | 12.9ms | 15.1ms | 22.3ms | 150 |
| Steady 4 | 12.9ms | 15.0ms | 21.5ms | 150 |
| Late | 12.5ms | 14.5ms | 15.6ms | 150 |

**Init timing**:
- Palm detect first resolution: 67.3ms (cold), settling to ~47-51ms
- Face detect first resolution: 20.3ms (cold), settling to ~7-10ms
- Camera resolution: 480x360

**Steady-state summary** (battery):
- Hand mean: ~8ms, p95: ~12.5ms
- Face mean: ~13ms, p95: ~15.5ms
- Max spikes decrease over time as GPU warms up and caches settle
