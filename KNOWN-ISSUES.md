# Known Issues

Things that are broken, flaky, or not-yet-understood. Add new entries at the top.
Include enough context that the next debugging session can pick up cold.

---

## BLOCKING: ONNX Runtime WebGPU does not work on iOS Safari

**Confirmed 2026-04-12.** iOS 26.4.1, Safari, Cloudflare Pages with correct COEP/COOP headers.

ONNX Runtime's WebGPU execution provider does not support iOS. The `initWebGPUBackend` function never reaches the pre-flight `navigator.gpu` check — it fails silently somewhere during module import or worker spawn. No error surfaces in Safari's console.

This is a Microsoft issue, not ours:
- [ORT #22776: Support iOS devices](https://github.com/microsoft/onnxruntime/issues/22776)
- [ORT #26827: Severe CPU/memory issues in Safari/WebKit 26](https://github.com/microsoft/onnxruntime/issues/26827)

**The fix is Phase 4: drop ONNX Runtime entirely and run inference with custom WGSL compute shaders.** See WORK-PLAN.md. This eliminates the 23MB WASM dependency, the SharedArrayBuffer/crossOriginIsolated requirement, and the iOS incompatibility in one move.

---

## BlazeFace detector: 2x faster but inherently jittery for position tracking

**Measured on M1 Max, Chrome 146, 640x480 camera:**

| Mode | Input | Mean latency | p95 |
|------|-------|-------------|-----|
| Detector (BlazeFace) | 128x128 | 8.2ms | 12.3ms |
| Landmark (Face Mesh) | 256x256 | 15.3ms | 26.2ms |

**The problem:** BlazeFace's 128x128 input gives it ~2 video pixel resolution per anchor step. Its bbox center oscillates +-0.04 in normalized coords even when the face is completely still (confirmed via `[roi]` debug logs, 2026-04-11). That's ~6px of jitter at 480p. The 6-keypoint spatial averaging (sqrt(7) noise reduction) helps but doesn't eliminate it because all 6 keypoints share the same quantization grid.

The landmark model at 256x256 has 2x the effective resolution and outputs 478 points from independent regressors, giving inherently more stable position estimates. It was designed for frame-to-frame tracking; BlazeFace was designed for detection ("is there a face? roughly where?").

**ROI tracking amplifies the problem:** When we crop around the last bbox and re-run BlazeFace on the crop, the detector's per-frame noise feeds back into the crop position, creating a feedback loop. The crop shifts 14px/frame from +-0.04 noise, which changes the model input, which changes the output, which shifts the crop. Confirmed visually via the CROP INPUT debug panel (orange box, lower right). EMA smoothing (factor 0.15) on the crop center dampens but doesn't eliminate the feedback.

**Current state:** ROI dropdown defaults to "Auto" (crop in center 90%, full-frame at edges, with hysteresis). Landmark mode is the right choice for head-coupled parallax where smoothness matters. Detector mode is useful for presence detection, spawn triggers, or latency-critical paths where +-6px jitter is acceptable.

**What would fix it for detector mode:** Landmark-driven ROI tracking (use the 478-point model's eye corners to compute next-frame crop, same as MediaPipe's internal architecture). That gives sub-pixel ROI updates from a high-res signal without running the full landmark model for position output. Not yet implemented.

**What we actually need:** A purpose-built detector that runs at ~512x512 input (enough resolution for full side-to-side head range without quantization stair-stepping) but outputs only ~6 anchor points, not 478 landmarks or 896 anchor boxes. Something between BlazeFace's "fast but coarse" and Face Mesh's "precise but heavy". Think: a tiny regression head trained to output face center + eye corners from a higher-res input, with no classification overhead. ~2ms inference, sub-pixel precision, no temporal smoothing needed. This model doesn't exist publicly. Training one is a Phase 4+ task.

---

## Intermittent: ball-toss loads but interaction is dead

**First seen:** 2026-04-11, ball-toss demo, WebGPU backend, Chrome on macOS.

**Symptoms:**
- Splash screen → Start Camera works
- 3D scene renders (cubes visible, tunnel visible)
- HUD is present
- ML workers all load successfully (palm, landmark x2, face detector, face landmark, blendshape)
- Console shows `[tracking] slots: _,_` firing on a 2s cadence (pipeline.js rate-limited logger) and actual `[palm] 1 detections` / `[new hand] slot 0` entries — so frames ARE flowing through the WebGPU pipeline
- BUT: scene does not respond to the user. Head parallax doesn't update, hand skeletons don't draw, pinch doesn't spawn balls. As if `currentHands` / `headRaw` never make it into `animate()`.
- Self-view lower right not showing (may be expected — depends on `selfView` localStorage flag)

**What we verified was fine:**
- `getUserMedia` succeeded, `video.srcObject` was set, `video.play()` returned (otherwise the pipeline couldn't have produced detections)
- `gpuReady === true` (otherwise `processWebGPUHands` would have early-returned)
- `wgpuVfcTick` was firing (otherwise no detection logs)
- All ORT warnings in the console were the harmless "some nodes not assigned to preferred EP" messages

**Suspected causes (not yet confirmed):**
1. `animate()` threw on its first run after `revealSceneUI()` and killed the rAF loop before it could render any tracked state. The scene stays on whatever Three.js drew during the first paint, so cubes are visible but nothing updates. This would match all symptoms.
2. A race between the page-load preload (`initWebGPUBackend(true)`) and the user-click path (`loadCameraAndModels` → `initWebGPUBackend()`). Specifically: the preload path sets `gpuReady = true` and calls `startWebGPUVfcLoop()` BEFORE `video.srcObject` exists. `wgpuVfcRunning` becomes true but `wgpuVfcChainActive` stays false (the rVFC registration is gated on `video.srcObject`). When the user-click path later runs, `initWebGPUBackend()` fast-returns via the `wgpuInitPromise` cache and calls `startWebGPUVfcLoop()` again, which IS supposed to register rVFC now that `srcObject` is live. Possible that the re-registration is racing `video.play()` resolution. But this theory doesn't explain why detections were still coming through — if the rVFC chain were dead, `processWebGPUHands` would never fire.
3. `handleHandResult` got called but `handleFaceResult` didn't — `headRaw` stays at (0,0,0), so head parallax appears frozen. Possible if the face detector is returning early (e.g. no detection in the first frames, full-frame fallback hasn't kicked in) and there's a stuck state somewhere. Doesn't explain hands not drawing though.

**How to debug when it happens again:**
1. Open DevTools BEFORE clicking Start Camera so we catch the very first error.
2. Check the actual red errors (not the ORT warnings).
3. In the console, manually inspect: `currentHands`, `gpuReady`, `wgpuVfcRunning`, `wgpuVfcChainActive`. Most diagnostic if run mid-failure.
4. Check whether `animate` is still in the rAF queue: `performance.now()` vs the FPS counter element — if FPS has frozen, the loop is dead.
5. Look for any uncaught promise rejection from `processWebGPUHands().then()` chains.

**Not reproducible on demand.** Seen once, did not reproduce on immediate retry. Likely a race condition, not a logic bug in a hot path.

---
