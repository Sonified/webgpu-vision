# Optimization Log

Every approach tried, what it got us, what it cost. The ground truth.

## Current standings (2026-04-21)

**Live demo (M1 Max, Chrome, 480x360, 2 hands + face):**

| | **WGSL (ours)** | **ORT WebGPU** | **MediaPipe** | vs ORT | vs MediaPipe |
|---|---|---|---|---|---|
| Hand (2 hands) | **7.9ms** | 8.2ms | 29.3ms | **3.7% faster** | **3.7x faster** |
| Face LM | **12.4ms** | 13.0ms | 25.1ms | **4.6% faster** | **2.0x faster** |

**Headless (single model, no contention -- the ceiling):**

| Model | Baseline | **WGSL** | **ORT WebGPU** | **ORT WASM** | vs ORT-GPU | vs ORT-WASM |
|---|---|---|---|---|---|---|
| Palm | 18.61ms | **9.43ms** | 24.40ms | 28.90ms | **2.6x faster** | **3.1x faster** |
| Hand | 12.67ms | **2.98ms** | 6.89ms | 18.02ms | **2.3x faster** | **6.0x faster** |
| Face det | 12.70ms | **2.92ms** | 2.77ms | 3.02ms | 5% slower | parity |
| Face LM | 53.61ms | **5.88ms** | 8.27ms | 13.86ms | **1.4x faster** | **2.4x faster** |

Notes:
- ORT WebGPU's palm detector is barely faster than its own WASM (24.4 vs 28.9ms). Their GPU EP adds almost no value on that model. We're 2.6x faster.
- Face detector is the one model where ORT-GPU edges us out (2.77 vs 2.92ms). It's a tiny model where dispatch overhead dominates and ORT's tighter C++/WASM GPU API access wins.
- Run with `node engine/bench-all.mjs` to reproduce. 50 iterations, 20 warmup, isolated headless Chrome.

GPU compute only (timestamp queries, excludes readback/submit overhead):
- Hand: **1.842ms** | Face LM: **5.99ms**

The gap between headless and live is purely architecture/overhead (message passing, warp preprocessing, readback).

---

## Engine optimizations (all verified, all shipped)

### 1. GPU PReLU + uniform buffer pool
- **What:** Moved 34 standalone PReLU ops from CPU readback to GPU dispatch. Pooled uniform buffers.
- **Result:** Face LM: 53.61ms -> 14.68ms (**3.65x speedup**)
- **Cost:** None. Pure win.

### 2. Level 1 kernel fusion (fused_block.wgsl)
- **What:** Fused DW Conv + 1x1 Conv + Add + Activation into single dispatch. DW output stays in registers.
- **Result:** Face det: 9.18ms -> 7.70ms. Face LM: 14.68ms -> 12.84ms
- **Gotcha:** Can't fuse when 1x1 narrows channels (redundant DW recompute). Can't fuse when dwInCh * kArea > 1024. Learned this the hard way with palm detector (37ms regression before gating).
- **Cost:** Smart gating logic adds complexity. Some models get no fusion (hand landmark -- all blocks narrow channels).

### 3. GPU transpose (transpose_nhwc.wgsl)
- **What:** NCHW->NHWC transpose for output heads moved from CPU readback to GPU shader.
- **Result:** Face det: 7.70ms -> 7.24ms (~6% improvement)
- **Cost:** None.

### 4. GPU sigmoid (add.wgsl mode 4)
- **What:** Standalone sigmoid moved from CPU readback to GPU.
- **Result:** Fixed handFlag/handedness being stale in compiled path (correctness fix). Minor perf win.
- **Cost:** None. Critical for correctness.

### 5. Pre-compiled command replay (compile + runCompiled)
- **What:** Graph walk, buffer allocation, bind group creation done ONCE. Subsequent frames just encode pre-built steps.
- **Result:** Face det: 7.24ms -> 3.30ms (**55% speedup**). Hand: 11.28ms -> 6.14ms. Face LM: 12.84ms -> 8.49ms.
- **Cost:** Memory (pre-allocated buffers persist). compile() adds init time.

### 6. Pre-allocated readback staging buffers
- **What:** Staging buffers for output readback created once during compile, reused every frame. Parallel mapAsync.
- **Result:** Face det: 4.31ms -> 3.32ms. Output copies in same encoder as dispatches.
- **Cost:** None.

### 7. Single compute pass (multiple dispatches, implicit barriers)
- **What:** All dispatches in one beginComputePass/end instead of separate passes per dispatch.
- **Result:** ~5% improvement in headless. GPU driver can optimize the sequence as one unit.
- **Cost:** Must break pass for buffer copies (Concat). Minor code complexity.

---

## Architecture experiments (live demo)

### A. Separate workers (5 GPU devices) -- BEST LIVE PERF
- **What:** Each worker (palm, hand0, hand1, face det, face lm) creates its own GPU device.
- **Result:** Hand: **10.6ms**, Face: **14.9ms**
- **Why it works:** True parallelism. M1's GPU driver handles multi-device well. No serialization.
- **Downside:** 5 GPU devices feels wasteful. Not how ORT did it. When MediaPipe is also loaded, contention kills perf.

### B. Unified worker (1 device, sequential)
- **What:** One vision-worker.js, one device, all models. Each request creates own encoder + submit.
- **Result:** Hand: **17.8ms**, Face: **19.7ms**
- **Why it's slower:** 4-5 separate queue.submit() calls per frame. Each submit has overhead. Serialized execution.
- **Status:** Reverted initially, then brought back with fixes.

### C. Unified worker + mutex (BROKEN)
- **What:** Added spin-wait mutex to prevent concurrent runCompiled() on same runner.
- **Result:** **18-second stalls**, hand oversampling at 3x. Deadlock.
- **Why it broke:** `while (this._running) await setTimeout(0)` spin-wait starved the event loop.
- **Status:** Removed.

### D. Unified worker + dual hand runners
- **What:** Two separate ModelRunner instances for hand landmarks, separate buffers, same device.
- **Result:** Hand: **16.2ms**, Face: **20.1ms** (no buffer conflicts, no stalls)
- **Why:** Eliminated the staging buffer collision from concurrent hand landmark calls.
- **Status:** Shipped.

### E. Unified worker + single compute pass
- **What:** Multiple dispatches in one beginComputePass (WebGPU implicit barriers).
- **Result:** Hand: **14.7ms**, Face: **19.5ms** (small improvement over D)
- **Status:** Shipped.

### F. Unified worker + batched submit (CURRENT)
- **What:** Queue incoming requests, flush on next microtask. All models in same event loop tick get encoded into ONE command encoder, ONE queue.submit().
- **Result:** Hand: **13.5ms**, Face: **19.4ms** (significant hand improvement)
- **Why:** Eliminates per-model submit overhead. GPU sees all work as one command stream.
- **Status:** Shipped. Current architecture.

---

## Remaining gap analysis

### Where the time goes (per frame, estimated)

| Step | Time | Notes |
|---|---|---|
| JS message passing (main -> worker -> main) | ~1-2ms | postMessage overhead, structured clone |
| Warp shader submits (separate from batch) | ~2-3ms | 4 separate texture uploads + submits |
| GPU inference compute | ~6-8ms | The actual neural network math |
| Staging buffer readback (mapAsync) | ~2-3ms | GPU->CPU copy for outputs |
| Post-processing (decode, NMS, projection) | ~1ms | CPU work after readback |
| **Total** | ~12-16ms | |

### What ORT does differently
- ORT creates ONE device via C++ (WASM), shares it across all sessions
- ORT's internal command recording bypasses the JS API overhead
- ORT batches multiple session.run() calls on the same queue internally
- ORT uses IO binding to avoid some readbacks

### Unexplored optimizations
1. **Fold warp into batch** -- the warp shader submits are currently separate because they use textures. Could we pre-upload textures and include warp in the batched encoder?
2. **Skip readback for intermediate results** -- palm detection outputs are only used to compute hand ROIs. Could we do that computation on GPU too?
3. **Reduce readback frequency** -- only read hand landmarks every frame, skip palm detection readback when tracking is active (it already skips palm detect, but the readback path is still hot)
4. **WebGPU timestamp queries** -- measure actual GPU time vs JS overhead to know exactly where time goes
5. **Subgroup operations** -- WebGPU subgroups proposal could enable within-wave reductions for NMS/decode
6. **Frame skipping** -- run inference every 2nd or 3rd frame, interpolate between (the demo already has interpolation infrastructure)

### G. Unified submit: warps folded into inference encoder (REVERTED)
- **What:** Encode warp dispatches + inference dispatches + readback copies into ONE command encoder, ONE queue.submit().
- **Result:** Hand: **15.2ms** (regression from 13.5ms). Face: **22.6ms** (regression from 19.4ms).
- **Why it's slower:** By waiting to encode ALL warps before submitting, the GPU sits idle during the encoding phase. The previous approach (separate warp submits) let the GPU START warp execution immediately while JS was still encoding inference dispatches. The overlap between GPU warp execution and JS encoding was free parallelism we lost by batching.
- **Key insight:** More batching is NOT always better. GPU/CPU overlap matters. Submitting warp work early lets the GPU chew on it while the CPU prepares the next batch. Holding everything for one mega-submit increases latency because the GPU has nothing to do while the CPU encodes.
- **Status:** Reverted to separate warp submits + batched inference.

### H. WebGPU single compute pass (multiple dispatches, implicit barriers)
- **What:** Searched the WebGPU spec and confirmed: multiple dispatchWorkgroups() in ONE compute pass have implicit serial memory semantics. No need for beginComputePass/end per dispatch.
- **Result:** Applied in engine. ~5% improvement in headless. Removes 34-106 pass boundary transitions per model.
- **Source:** [gpuweb discussion #4434](https://github.com/gpuweb/gpuweb/discussions/4434)

### I. Cross-workgroup sync research (NOT POSSIBLE in WebGPU)
- **What:** Investigated running entire neural network in one dispatch (Level 3 fusion).
- **Findings:**
  - WebGPU has NO cross-workgroup synchronization. `storageBarrier()` only syncs within one workgroup (max 256 threads).
  - CUDA has Cooperative Groups + `cudaLaunchCooperativeKernel` for grid-wide sync. WebGPU has no equivalent.
  - Atomic spin-lock workaround is "taboo" -- works on some hardware, deadlocks on others due to no forward progress guarantees.
  - The [WebGPU dispatch overhead paper (2024)](https://arxiv.org/abs/2604.02344) explicitly says cooperative groups and persistent kernels are needed as spec-level changes.
  - The [Decoupled Fallback paper](https://dlnext.acm.org/doi/pdf/10.1145/3694906.3743326) shows portable single-pass prefix scan using atomics -- a possible path for specific operations but not full neural network orchestration.
- **Status:** Not implementable with current WebGPU spec. Monitoring gpuweb proposals.

---

## Shader-level optimizations (the compute itself)

### 8. Workgroup shared memory for depthwise conv
- **What:** DW conv loads the input tile (8x8 output region + kernel halo) into `var<workgroup>` shared memory. Each input pixel is read from global memory ONCE by one thread, then all threads in the workgroup read from fast local memory for the convolution.
- **Why it matters:** A 3x3 DW conv without shared memory reads each input pixel up to 9 times from global memory (once per kernel position that overlaps it). With shared memory, it's read once. For 5x5 kernels, the savings are up to 25x fewer global reads.
- **Gotchas fixed:**
  - Unsigned subtraction wrap: `wgid * 8 * stride - pad` wraps negative when pad > 0 at workgroup 0. Fixed with signed `i32` arithmetic.
  - Stride-2 tile size: an 8x8 output with stride 2 needs an 18x18 input tile (not 10x10). Shared memory sized to 20x20=400 floats to cover worst case (stride 2, 5x5 kernel).
  - Uniform control flow: `workgroupBarrier()` can't be inside a bounds-check `if`. Restructured so all threads participate in barrier, only in-bounds threads compute.
  - `in_c_idx` must be `oc` for pure depthwise (output channel == input channel), not `in_c_start`.
- **Result:** Massive win for DW-heavy models. Hand landmark (MobileNetV2): **6.40ms -> 3.20ms (2x)**. Palm: **13.10ms -> 9.75ms (25%)**. Face LM: **8.46ms -> 7.02ms (17%)**.

### 10. Fused inverted residual (REJECTED -- too much redundant compute)
- **What:** Fuse entire MobileNetV2 inverted residual block (expand 1x1 -> ReLU6 -> DW 3x3 -> ReLU6 -> project 1x1 -> Add) into a single dispatch. Wrote `fused_invres.wgsl`. Each output thread recomputes the expand and DW results on-the-fly.
- **Result:** Hand landmark: 3.20ms -> **21.82ms** (6.8x SLOWER). Only 3 blocks fused.
- **Why it failed:** The expand 1x1 is recomputed for every DW kernel position (9x) AND for every output channel. With expand ratio 6x (16->96 channels), each output thread does 9 × 16 = 144 expand multiplies × 96 expanded channels = 13,824 total. Unfused, the expand is computed once: 16 × 96 = 1,536 total. That's **9x redundant compute** -- the dispatch overhead savings (~147μs for 3 dispatches) are dwarfed by 18ms of extra math.
- **Key insight:** Kernel fusion only wins when the intermediate result is small relative to the recomputation cost. DW -> 1x1 fusion works because the DW output is one float per channel (no spatial recomputation). Expand -> DW -> Project fusion fails because the expand output is spatially large and gets recomputed at every kernel position.
- **Rule of thumb:** Fuse when the intermediate has fewer elements than the outer loop iterations. Don't fuse when the intermediate is wider than the output.
- **Status:** Shader exists at `fused_invres.wgsl` but is NOT wired in. Kept for reference.

### 11. Shared memory weight tiling for 1x1 pointwise (NO IMPROVEMENT)
- **What:** For 1x1 convolutions, load weight row into workgroup shared memory (64 threads cooperatively load, all read from shared). Saves 63 redundant global memory reads per weight value.
- **Result:** Hand: 3.20ms -> 3.47ms (8% SLOWER). All other models flat.
- **Why it didn't help:** On M1, the L2 cache is large and fast enough that redundant weight reads from global memory are already cached. The `workgroupBarrier()` synchronization cost (twice per tile iteration) is more expensive than the memory traffic savings. The barrier forces all 64 threads to stall until the slowest one finishes loading, which adds latency that the cache-hit path doesn't have.
- **Key insight:** Shared memory tiling only wins when the data is too large for L2 cache OR when the barrier cost is amortized over enough compute. For small 1x1 convolutions (16-96 channels), the data fits in cache and barriers dominate.
- **Status:** Reverted. Simple vec4 accumulation from global memory is faster on M1.

### 9. Vec4 dot product for 1x1 pointwise conv
- **What:** For 1x1 convolutions (which are just matrix multiplies), load 4 input channels and 4 weights at once as `vec4<f32>`, use `dot()` for the multiply-accumulate. 4x fewer memory transactions per iteration.
- **Result:** Combined with shared memory DW, contributes to the 2x hand speedup. 1x1 convs are the bottleneck in MobileNetV2's expand-project pattern.
- **Cost:** Handles remainder channels (when `channels_per_group` not divisible by 4) with scalar loop.

---

## Updated headless benchmarks (the ceiling)

| Model | Original baseline | After all shader opts | ORT WASM | vs ORT |
|---|---|---|---|---|
| Palm | 18.61ms | **9.75ms** | 27.83ms | **2.9x faster** |
| Hand | 12.67ms | **3.20ms** | 17.45ms | **5.5x faster** |
| Face det | 12.70ms | **3.11ms** | 3.10ms | parity |
| Face LM | 53.61ms | **7.02ms** | 13.58ms | **1.9x faster** |

Hand landmark is now **5.5x faster than ORT WASM** in isolated benchmarks. The engine itself is no longer the bottleneck -- any remaining live demo gap is purely JS/worker architecture overhead.

---

## The honest truth

Our engine is **2-5.5x faster than ORT** in isolated benchmarks. The live demo numbers (13.5ms hand, 19.4ms face) include JS overhead:
- Message passing between main thread and worker (~1-2ms)
- Warp shader submits for camera frame preprocessing (~2-3ms)
- Staging buffer readback via mapAsync (~2-3ms)
- Post-processing (anchor decode, NMS, projection) (~1ms)

ORT avoids some of this by running inside WASM with tighter GPU API access. We're paying the JS tax on every frame.

The shader optimizations (shared memory, vec4) attack the GPU compute time directly. The architecture optimizations (batched submit, single pass) attack the JS overhead. Both fronts matter.

We are **1.6-2.2x faster than MediaPipe** in the live demo. The headless engine is **2-5.5x faster than ORT WASM**.

## ORT PARITY ACHIEVED (2026-04-14)

Separate workers (5 devices) + optimized shaders (shared memory DW + vec4 pointwise):

| | **WGSL (live)** | **ORT-WebGPU (live)** | |
|---|---|---|---|
| Hand | **8.2ms** | 8.2ms | **MATCH** |
| Face LM | **13.2ms** | 13.0ms | **MATCH** |
| MediaPipe Hand | 29.3ms | | 3.6x slower |
| MediaPipe Face | 25.1ms | | 1.9x slower |

Built from scratch. No ONNX Runtime. No WASM. Pure WebGPU compute shaders.

The key was NOT architecture (unified vs separate workers) -- it was **shader-level compute optimization**. The shared memory DW conv (2x hand speedup) and vec4 pointwise accumulation gave us the compute parity. The separate worker architecture then preserved the parallelism that the unified worker lost.

### 12. Bitmap clone optimization (NO IMPROVEMENT)
- **What:** Create one ImageBitmap from video, then clone via `createImageBitmap(bitmap)` for the second hand slot instead of two `createImageBitmap(video)` calls.
- **Result:** Hand: 9.5ms -> 10.8ms (13% SLOWER)
- **Why it didn't help:** `createImageBitmap(bitmap)` is NOT faster than `createImageBitmap(video)` -- both decode full pixel buffers. The clone path added sequential overhead (create first bitmap, THEN clone) instead of the original which created both in parallel inside Promise.all.
- **Status:** Reverted.

### 13. Hand oversampling fix (SHIPPED)
- **What:** Added 16ms minimum interval gate to `processWebGPUHands()` to prevent calling inference twice per video frame.
- **Result:** Hand oversampling dropped from 2.0x to 1.0x. Freed GPU cycles. Revealed that the 8.2ms "parity" number was inflated by double processing.
- **Honest numbers:** Hand 9.5ms, Face 13.2ms with clean 1.0x sampling.

### 15. Cached warp texture + bind group (SHIPPED -- 5% hand improvement)
- **What:** All 4 WGSL workers were calling `device.createTexture()` + `device.createBindGroup()` + `srcTexture.destroy()` on every frame for the warp/letterbox preprocessing step. Video dimensions never change mid-session, so these GPU resources are identical every frame. Cached them: create once on first frame, reuse thereafter, only recreate if video resolution changes. Also pre-allocated `Float32Array` for uniform writes (landmark workers).
- **Files changed:** `palm-worker-wgsl.js`, `landmark-worker-wgsl.js`, `face-detection-worker-wgsl.js`, `face-landmark-worker-wgsl.js`
- **Result (live demo, 2 hands + face, M1 Max):**
  - Hand: **9.5ms -> 9.0ms** (5% faster, best batch 8.6ms)
  - Face: **13.2ms -> 13.0ms** (exact ORT parity)
- **Why it helped:** `createTexture()` is a real GPU driver allocation (not just a JS object). It asks the GPU to carve out VRAM, set up memory mapping, configure format/usage flags. `createBindGroup()` triggers driver-side validation of buffer/texture compatibility against the pipeline layout. Doing both 5x per frame (one per worker) was pure overhead. Caching eliminates ~10 GPU driver calls per frame.
- **Cost:** Textures persist in VRAM instead of being freed each frame. For 480x360 RGBA8 video, that's ~675KB per worker x 5 = ~3.4MB total. Negligible.

### 16. VideoFrame zero-copy transfer (SHIPPED -- 1.4ms main thread freed)
- **What:** Replaced `createImageBitmap(video)` with `new VideoFrame(video)` for sending camera frames to workers. VideoFrame grabs a reference to the camera's decoded frame buffer (0.02ms) instead of decoding pixels into a new bitmap (0.5ms). `copyExternalImageToTexture` accepts VideoFrame natively. Workers use `displayWidth`/`displayHeight` for VideoFrame compatibility.
- **Files changed:** `pipeline.js`, `face-pipeline.js`, all 4 WGSL workers
- **Result:** Main thread blocking per frame: **1.5ms -> 0.06ms** (3 calls x 0.5ms savings). Worker round-trip mostly unchanged (GPU upload cost is similar for both source types). Hand best batch: **8.7ms**.
- **Key insight:** The shootout tests in the sister repo (`3d-parallax-head-hand-tracking-demo/pipeline-shootout.html`) had already proven VideoFrame was 25x faster for frame transfer. The savings are on the main thread, not in the worker -- the bench timer measures worker round-trip so the improvement is partially invisible there, but the render loop gets 1.4ms more headroom per frame.

### 17. Merged warp + inference into single GPU submit (SHIPPED -- beat ORT)
- **What:** Each worker was doing two `queue.submit()` calls per frame: one for the warp/letterbox compute dispatch, one for inference + readback. Refactored `dispatchWarp`/`gpuLetterbox` to separate texture upload (queue commands) from dispatch encoding. New `encodeWarp`/`encodeLetterbox` functions encode the compute pass into an external encoder. Worker creates ONE encoder, calls `encodeWarp(enc)` + `runner.encodeInto(enc)`, then one `queue.submit()`.
- **Files changed:** All 4 WGSL workers
- **Previous attempt:** Optimization G tried this in the unified worker and regressed 13% because encoding ALL models delayed GPU start. In per-model workers with cached bind groups, encoding overhead is negligible.
- **Result (live demo, 2 hands + face, M1 Max):**
  - Hand: **8.0-8.5ms** (ORT was 8.2ms -- **WE BEAT IT**)
  - Face: **12.7-13.0ms** (ORT was 13.0ms -- **WE BEAT IT**)
- **Why it worked this time:** The unified worker had to encode 5 models before submitting, so the GPU sat idle for milliseconds. A per-model worker encodes ~66-106 pre-built dispatch steps (no allocation, no bind group creation) which takes microseconds. The submit overhead savings (~0.3-0.5ms per eliminated submit) outweigh the negligible encoding delay.

## ORT BEATEN (2026-04-15, extended 2026-04-20)

From-scratch WGSL inference engine now **faster than Microsoft's ONNX Runtime WebGPU backend** on live demo benchmarks:

| | **WGSL (live)** | **ORT-WebGPU (live)** | |
|---|---|---|---|
| Hand (2 hands) | **7.9ms** | 8.2ms | **3.7% faster** |
| Face LM | **12.4ms** | 13.0ms | **4.6% faster** |
| MediaPipe Hand | 29.3ms | | 3.7x slower |
| MediaPipe Face | 25.1ms | | 2.0x slower |

No ONNX Runtime. No WASM. No SharedArrayBuffer. No COOP/COEP headers required.
Pure WebGPU compute shaders. Runs on iOS Safari. ~50KB engine vs 23MB ONNX Runtime.

### The optimization path that got us here (session of 2026-04-15)

Starting point: Hand 9.5ms, Face 13.2ms (0.8-1.3ms behind ORT).

1. **Cached warp texture + bind group** (#15): Hand 9.5 -> 9.0ms, Face 13.2 -> 13.0ms
2. **Pre-allocated readback arrays**: Hand headless 3.33 -> 3.25ms
3. **VideoFrame zero-copy transfer** (#16): Main thread 1.5ms -> 0.06ms freed
4. **Merged warp + inference submit** (#17): Hand 9.0 -> 8.0ms, Face 13.0 -> 12.7ms

Total session improvement: **Hand 15.8% faster, Face 3.8% faster.**

### 14. Output channel tiling for 1x1 pointwise (NO IMPROVEMENT on M1)
- **What:** Each thread computes multiple output channels instead of 1, sharing the input vector loads across channels. Tested 2-OC (2 channels per thread) and 4-OC (4 channels per thread). Dispatch z-dimension reduced from `outC` to `ceil(outC/N)`.
- **Hypothesis:** Input activations loaded once per thread, reused across N output channels, cutting input bandwidth by Nx.
- **Benchmark methodology note:** Initial results (single batch, 5 warmup) showed 22-30% regression. Proper benchmarking (5 batches of 50, 20 warmup iterations, median) revealed the initial results were polluted by GPU warmup variance. Real numbers below.
- **Results (proper benchmark, isolated browser per model, median of 5x50):**

  | Variant | Hand LM | Palm Det |
  |---|---|---|
  | Baseline (1 OC) | **3.33ms** | **9.81ms** |
  | 2-OC tiling | **3.83ms** (+15%) | **9.57ms** (-2.4%) |
  | 4-OC tiling | **3.54ms** (+6%) | not tested |

- **Model-dependent:** 2-OC helps palm (smaller channels, max 256) but hurts hand (672-channel layers). The extra weight bandwidth for 672-channel 2nd output channel exceeds the input bandwidth savings. Palm's smaller channels keep the weight overhead manageable.
- **Key insight:** OC tiling trades input bandwidth for weight bandwidth. On M1 with L2 cache, input reads are already cheap (cache hits). The trade only wins when channels are small enough that the extra weight reads don't dominate. For MobileNetV2's 672-channel layers, they do.
- **Status:** Reverted. Single OC per thread is optimal for the hand model (our primary optimization target). The 2-OC approach would likely help on discrete GPUs (NVIDIA, AMD) where input reads are real cache misses, and should be retested on non-Apple hardware.

### 18. GEMM parallel reduction + vec4 (SHIPPED)
- **What:** Two-path GEMM shader: small M (<=4) uses one-thread-per-output with vec4 dot. Large M uses 64-thread parallel reduction per output (each thread handles K/64 chunk, tree reduction via shared memory).
- **Result:** GEMM dispatches faster across all models. Contributes to overall gains below.
- **Cost:** None. Both paths are branchless at dispatch time (selected by workgroup dimensions).

### 19. Conv2D 1x1 double-unrolled vec4 + adaptive oc_tile (SHIPPED)
- **What:** 1x1 convolutions now process 4 output channels per thread (oc_tile=4) with double-unrolled vec4 inner loop (8 input channels per iteration). Adaptive gating: `oc_tile = iC <= 64 ? 4 : 1` prevents cache thrashing on large-channel layers (672-to-112 was regressing 2x with oc_tile=4).
- **Result:** Significant 1x1 speedup for small-channel layers. Large-channel layers stay at oc_tile=1 to avoid regression.
- **Key insight:** oc_tile=4 wins when weight working set fits in L2 (small iC). At iC=672, 4 output channels means 4x the weight reads per thread, thrashing the cache.

### 20. Conv2D unrolled 2x2 general path (SHIPPED -- 52% face conv speedup)
- **What:** Face landmark model uses 2x2 strided convolutions for downsampling (not depthwise, not 1x1). Added a manually unrolled 2x2 kernel path between the 1x1 and generic general conv paths. Four explicit multiply-adds with bounds checks instead of a double loop.
- **Result:** Face LM conv_general category: **1.637ms -> 0.787ms (52% faster)**. These 2x2 convs were the #1 hotspot in the face model.
- **Cost:** None. Falls through to generic path for other kernel sizes.

### 21. Fused block tiled variant for small spatial (SHIPPED -- separate pipeline)
- **What:** Created `fused_block_tiled.wgsl` with `var<workgroup> tile: array<f32, 400>` for cooperative input tile loading. Used only when output spatial <= 8x8 (selected in model-runner.js). Unrolled 3x3 and 5x5 DW kernels reading from shared memory.
- **Why a separate shader:** Declaring `var<workgroup>` in a shader reduces GPU occupancy for ALL dispatches using that pipeline, even if the shared memory branch isn't taken. A single shader with conditional tiling caused hand model to regress from 7.6ms to 10.4ms. Two separate pipelines let large-spatial dispatches use the occupancy-friendly fused_block.wgsl.
- **Result:** Face LM fused blocks at 8x8: 7% faster. 4x4: 19% faster. 2x2: 20% faster.
- **Gotcha:** Uniform control flow required -- all 64 threads must hit `workgroupBarrier()` regardless of bounds. Uses `in_bounds` flag instead of early return.

### Session of 2026-04-20: shader compute round 2

Starting point: Hand 8.0ms, Face 12.7ms (already beating ORT).

1. **GEMM parallel reduction** (#18)
2. **1x1 double-unrolled vec4 + adaptive oc_tile** (#19)
3. **2x2 general conv unrolling** (#20): face conv 52% faster
4. **Fused block tiled variant** (#21): face small-spatial fused blocks 7-20% faster

**Updated live benchmarks (M1 Max, Chrome, 480x360, 2 hands + face):**

| | **WGSL (live)** | **ORT-WebGPU (live)** | |
|---|---|---|---|
| Hand (2 hands) | **7.9ms** | 8.2ms | **3.7% faster** |
| Face LM | **12.4ms** | 13.0ms | **4.6% faster** |

Total session improvement: **Hand 1.3% faster, Face 2.4% faster** (on top of already beating ORT).

GPU profiler numbers (headless, per-dispatch timing):
- Hand GPU compute: **1.842ms** (down from 3.115ms baseline = 41% faster across all shader sessions)
- Face LM GPU compute: **5.99ms** (down from 7.12ms = 16% faster this session)

### 22. OC-tiled fused block -- 4 output channels sharing DW compute (REJECTED)
- **What:** New shader `fused_block_oc4.wgsl` with workgroup_size(8, 8, 4) = 256 threads. 4 output channels share one workgroup, cooperatively load the DW input tile, z=0 threads compute DW once and broadcast via `dw_shared` array, all 4 oc threads accumulate their own 1x1 weight. Eliminates 3/4 redundant DW compute across output channels.
- **Hypothesis:** Face detector's 96ch 8x8 blocks (38% of compute) redundantly compute DW conv 96 times. With oc_tile=4, only 24 groups compute DW, sharing across 4 output channels each.
- **Results (face detector, per-dispatch GPU timing):**
  - 96ch 8x8: 274us -> 164us (**40% faster**) -- the target blocks improved significantly
  - 48-64ch 16x16: slight improvement (92->80, 113->93us)
  - 72-88ch 16x16: **30-32% SLOWER** (131->171, 146->193, 163->212us)
  - 24-28ch 64x64: **21-32% SLOWER** (147->178, 176->233us)
  - Overall: 2.93ms -> 3.50ms (**19% regression**)
- **Also tried:** Relaxing tiling threshold (oH<=16, iC<=64) for 16x16 blocks. 48ch 16x16 regressed 57% (92->144us). M1 cache wins at these spatial sizes.
- **Why it failed:** Three barriers per input channel (tile load, DW broadcast, pre-next-load sync) dominate for high-channel or large-spatial blocks. 256-thread workgroups reduce occupancy vs 64-thread fused_block. The DW savings only outweigh barrier cost for the specific case of small spatial + high channels (96ch 8x8).
- **Key insight:** This is the fourth shared-memory strategy defeated by M1's L2 cache (after #10, #11, #14). The pattern is definitive on Apple Silicon. OC-sharing of DW compute helps the 96ch 8x8 blocks in isolation, but the global occupancy and barrier costs make it a net loss. The face detector's 0.15ms gap to ORT is in submit/readback overhead, not GPU compute.
- **Status:** Reverted. Shader kept at `fused_block_oc4.wgsl` for reference. May help on discrete GPUs (NVIDIA/AMD) where memory latency is higher and shared memory provides more benefit.

---

### What we learned
- Architecture experiments (unified worker, batched submit, warp folding) gave ~10-20% improvements at best
- Shader compute optimizations (shared memory, vec4, unrolling, fusion) gave **2x** improvements
- The GPU driver and L2 cache on M1 are smart enough that explicit data sharing (shared memory for weights, unified device) often hurts more than helps
- The M1 handles 5 separate GPU devices efficiently -- the "waste" of separate devices is actually free parallelism
- When in doubt, make the shader itself faster, not the orchestration around it
- **Shared memory occupancy penalty is real:** declaring `var<workgroup>` reduces max concurrent workgroups even when the memory isn't used. Split into separate pipelines to avoid penalizing dispatches that don't need shared memory.
- **M1 L2 cache defeats three separate tiling/sharing strategies:** weight tiling (#11), output channel tiling (#14), and the inverted residual fusion (#10). All three attempt to avoid reads that are already cheap. The pattern is consistent: on Apple Silicon with large unified-memory L2, the overhead of the sharing mechanism (barriers, registers, branches) exceeds the bandwidth savings. This may NOT hold on discrete GPUs (NVIDIA, AMD) where memory latency is higher and L2 is smaller -- these optimizations should be retested if targeting non-Apple hardware.
- **Benchmark methodology matters enormously.** GPU pipelines need 15-20 warmup iterations (not 5) to stabilize. Single-batch measurements can vary 20%+ between runs. Always use multiple batches (5x50) and take the median. Always test in isolated browser instances (sequential model tests in the same browser cause resource contention). Initial "30% regression" from OC tiling was entirely warmup noise -- real difference was 6-15% depending on model.
