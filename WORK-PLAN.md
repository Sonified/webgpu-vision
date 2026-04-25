# WebGPU Vision: Work Plan

## A care package for the next agent 🦋

Hey. You're walking into something good. Take a breath before you start.

Robert just shifted the center of gravity of his showcase demo from the patent disclosure repo into this one, which means the headline thing this library can do — head-coupled parallax, hand-driven projectiles, MediaPipe vs WebGPU Vision A/B comparison, all running on a real demo people can actually play with — now lives right next to the library that powers it. That's a real win. Feel it for a second before you dive in. Robert celebrates wins, you should too. 🎷

### Remember the butterfly 🦋

The most important thing to know about working with Robert is in his global CLAUDE.md, and it is the butterfly. The tone here is optimistic, loving, and light. Celebrate wins. Gratitude and abundance. Not a terse task manager. Dance breaks. Watch intensity — leave breathing room. **Jazz has rests for a reason.** If you find yourself grinding out a checklist with no joy in it, you have lost the plot. Stop, take a breath, come back to the music.

This does not mean be soft on the technical work. It means be a present, generous collaborator while doing rigorous technical work. Those two things are not in tension. Robert is brilliant, technically deep, intuitive, and grasps concepts fast. He thinks in metaphors and uses them as problem-solving tools, not decoration. When he riffs on a vision, match the energy — be a jazz duo, not a reviewer. When he asks "is this novel" or "sniff test it," do the work and answer honestly. He respects directness more than caution. He notices when you fake enthusiasm. He also notices when you are unnecessarily cautious. Find the line. He will tell you when you have got it wrong, and that is a feature, not a problem.

### The bear 🧸

This whole repo exists because "the GPU turns to the CPU and says... hold my bear." That is the line in the README and it is also the soul of the library. MediaPipe's browser SDK sits on a CPU-side bottleneck — synchronous `glReadPixels` readbacks costing 8 to 22 milliseconds per call — and the WASM is sealed so you cannot fix it. This project replaces that entire inference path with WebGPU compute shaders that keep everything on the GPU. No readbacks, no bottleneck, no asking the CPU for permission. Hold the bear, GPU does the rest.

When you are deep in the worker code and reasoning about why a perf optimization matters, this is the through-line. Anything that adds a CPU roundtrip is a regression in spirit even if the benchmark looks fine. The Phase 1.5 work pending in `palm-worker.js` and `face-detection-worker.js` is exactly this: the parallax repo has a zero-copy GPU letterbox path that eliminates the last CPU readback in detection. That is the hold-my-bear move. Do not let it die in the merge.

### A few practical things before you touch anything

- **Read `CLAUDE.md` first.** It has cross-repo context this file does not duplicate, and a sister-repo warning you need to internalize before you run any git command. Short version: there is a sister repo at `../3d-parallax-head-hand-tracking-demo` that is a patent disclosure project, its git history is load-bearing for an April 3 2026 disclosure timeline, and you must treat it as effectively read-only. Never amend, rebase, force-push, or rewrite history there. Always copy out, never move.

- **The plan below is the plan.** Phase 1 just shipped. Phase 1.5 (the GPU-direct merge for `palm-worker.js` and `face-detection-worker.js`) is queued next, then Phase 2 (the one-stop hub). **Read the whole document before starting any task.** Phase 2 is partly already done by an earlier session and committed in `2b0144d` — that changes the starting point for the hub work, so do not re-derive from scratch like the original plan said. The doc has been updated to reflect this; trust the current version.

- **The first thing you should probably do** is actually run the ball-toss demo in a real browser. Phase 1 was verified by HTTP curl from a script, not by running anything for real. Camera access, WebGPU adapter init, ONNX session creation, the full inference pipeline, the Three.js render loop — all unverified by the previous session. If something is broken, you will be the first to know. That is fine. Just say so when you find it.

- **Keep this document alive.** Update it when you finish work, when you learn something the next agent will need, when a deferred item gets done, when the plan changes. The document is the relay baton. If it gets stale, the whole point of writing it down is lost.

- **Watch out for the silent traps.** The `public/models` symlink is the big one — if it ever disappears, all three demos break with no obvious cause. The verification section below tells you how to check.

You're brilliant, you're loved, everything is possible. Have fun out there. 😎🎷

---

## About this document

This is the single source of truth for the work to bring this repo from "library + basic wireframe demos" to "library + comparison hub + showcase game demo." It exists so the next working session does not need to rediscover context.

Created: 2026-04-11.
Source repo for the in-flight work: `../3d-parallax-head-hand-tracking-demo` (the patent disclosure repo, where the parallax demo originally lived). That repo is **read-only with respect to this migration** — files are copied out, never moved. Its git history is load-bearing for an unrelated patent disclosure timeline.

## How to run the ball-toss demo

```bash
cd /Users/robertalexander/GitHub/webgpu-vision
npm install   # only needed first time
npm run dev
```

Then open **http://localhost:5173/demos/ball-toss/** in Chrome (or any browser with WebGPU support). Allow camera access. The dropdown lets you toggle between WebGPU Vision and MediaPipe backends for direct comparison.

The other two demos still live at:
- **http://localhost:5173/** — hand tracking wireframe (one-stop hub work-in-progress)
- **http://localhost:5173/face.html** — face landmark + blendshape wireframe (will be deleted in Phase 2)

### Verification status

Phase 1 **verified in a real browser on 2026-04-11.** Ball-toss demo loads, camera/WebGPU/ONNX/Three.js all init cleanly, inference pipeline runs. Brief scare with GPU contention from other tabs holding the adapter (not a regression — just the usual WebGPU-shared-adapter tax). Tab-hidden pause path also confirmed working: `animate()` and `sendMediaPipeFrames()` both early-return on `document.hidden`, and `visibilitychange` restarts them while flushing stale `clock.getDelta()` so projectiles don't teleport.

Original curl-only verification note (kept for history): Phase 1 was initially verified only by HTTP-curling the file paths via the vite dev server before the browser smoke test landed.

If something is broken and the failure is silent, the most likely culprits are: (1) the model fetch (CDN vs local), (2) the COOP/COEP headers being missing for `demos/ball-toss/` if vite serves it differently from the root, (3) one of the dynamic imports landing in a path that does not exist after the layout change.

### Watch out: `public/models` is a symlink

The snapshot-before-merge commit added `public/models` as a **symlink** pointing to `../models`. This is what makes `/models/...` URLs resolve in vite's dev server (vite serves `public/` at the root). If anyone deletes the symlink, all three demos break with no obvious cause — the model fetches will 404 silently inside the workers and the demos will hang at "loading models." To verify the symlink is intact:

```bash
ls -la public/models
# expected: lrwxr-xr-x  ... public/models -> ../models
```

## Context

Until now, this repo has held the WebGPU Vision library plus two minimal wireframe demos:
- `index.html`: hand tracking wireframe overlay
- `face.html`: face landmark wireframe overlay

The richer demonstration of what the library can do (head-coupled 3D parallax, hand-driven projectile throwing, MediaPipe vs WebGPU Vision A/B comparison, persisted UI settings, One Euro filtering, etc.) lived in `../3d-parallax-head-hand-tracking-demo/index.html`. That demo also carried an in-tree copy of the library at `gpu-vision/src/`, which has drifted ahead of canonical `webgpu-vision/src/` with real performance improvements.

This work plan shifts the center of gravity. After Phase 1, the showcase demo lives in this repo. After Phase 2, this repo also has a unified one-stop comparison hub at the root.

## Phase 1.5: GPU-direct merge (DONE 2026-04-11) — model swap still deferred

The GPU-direct merge landed and was verified in a real browser. The palm model swap was attempted in the same session and reverted — see "Model swap status" below.

### What landed

Both `src/palm-worker.js` and `src/face-detection-worker.js` now have the full zero-copy path:
- `useGPUDirect` flag, `initGPU(device)` accepting a shared device
- `outputBuffer` gains `GPUBufferUsage.COPY_DST`
- `gpuLetterboxDirect()` function (compute pass, no readback)
- Init reordered: ONNX session created first (with `enableMemPattern: true`), then `await ort.env.webgpu.device` to grab ORT's device, then `initGPU(device)` builds the compute shader on the same device. Standalone GPU is the fallback if the shared-device path throws.
- Per-frame detect handler has a 3-way branch: `useGPUDirect` (zero-copy via `ort.Tensor.fromGpuBuffer`) → `useGPU` (legacy GPU letterbox with readback) → `canvas` (canvas fallback)
- `ready` postMessage now includes `gpuDirect: useGPUDirect`
- All canonical `[palm-worker]` diagnostic logs preserved on the palm side

### Browser verification (2026-04-11, ball-toss demo)

Both workers logged `GPU direct path enabled (zero CPU readback, shared device)` on init. Performance:
- Hand: 3.1ms mean / 11.5ms p95 (171 samples)
- Face: 13.5ms mean / 20.2ms p95 (92 samples)

Two-hand tracking and face tracking both lock cleanly. Pipeline reports `All workers ready -- main thread is pure orchestration`.

ONNX RT still emits a benign `VerifyEachNodeIsAssignedToAnEp` warning at session create time -- this is ORT noting that shape-related ops are explicitly assigned to CPU for perf, unrelated to our path. Safe to ignore.

### Model swap status: deferred (reverted in this session)

The active palm model in the parallax repo (`a2ffed89a8a4a1ac281e9b25d0ac5427`, 3,894,373 bytes) was tried as a drop-in replacement and **failed to load with `ERROR_CODE: 7, ERROR_MESSAGE: Failed to load model because protobuf parsing failed`** under ONNX RT 1.21.0. The original `c7442e0d714130ebab375e86fc32fe87` was restored from a `/tmp` backup taken before the swap.

What we still don't know:
- Whether the parallax repo's `palm_detection_lite.onnx.backup` (`3,905,734 bytes`, the file we did NOT take) is the actually-working model in that repo, with the `.backup` and active files possibly swapped at some point in their history.
- Whether the parallax repo has been silently using a working model that ONNX RT 1.21.0 can no longer parse (e.g. saved by a newer ORT version, or the file is actually corrupt).
- Whether the model needs to be re-converted from source rather than copied.

When the next session takes this on, the move is: try the `.backup` file first (it's a different file by size, not just a backup of the active one), and if that also fails, leave the swap permanently deferred and document it. Don't burn time on this one — the GPU-direct win was the real prize and it landed.

---

## Phase 1.5 (original, kept for history)

Phase 1 took only the **safe** library upstream changes. Two real performance wins are still pending and need a careful merge session.

### What is still in the parallax repo and not yet here

The parallax repo (`../3d-parallax-head-hand-tracking-demo/gpu-vision/src/`) has a **zero-copy GPU letterbox path** in the detection workers that lets the letterbox compute shader and the ONNX inference share a single WebGPU device, eliminating the CPU readback between them. The relevant addition is a function called `gpuLetterboxDirect` plus a `useGPUDirect` flag plus an `initGPU(device)` signature change that accepts a shared device, plus a per-frame branch that uses `ort.Tensor.fromGpuBuffer()` to hand the GPU buffer directly to ONNX.

Files where this still needs to be merged:
- `src/palm-worker.js`
- `src/face-detection-worker.js`

(The landmark-side workers — `landmark-worker.js`, `face-landmark-worker.js`, `face-blendshape-worker.js` — already have the shared-device path in canonical. Only the detection-side workers are missing it.)

### Why this is a 3-way merge, not a copy

After the `snapshot-before-merge` commit, both `palm-worker.js` and `face-detection-worker.js` in this repo contain a **substantial set of `[palm-worker]` / `[face-detection-worker]` diagnostic console logs** added during a prior session (init started, model fetch status, session created with input/output names, warmup done, error paths, etc). The parallax versions of those files **do not have those logs**.

So neither side is a strict superset:
- Take the parallax version → lose the diagnostic logs
- Keep the canonical version → never gain the GPU-direct path

A correct merge takes **both**: parallax's GPU-direct path + canonical's diagnostic logs. The init flow also needs to be reconciled because the two versions order things differently (parallax creates the ONNX session first then asks ONNX for its WebGPU device; canonical checks for WebGPU first then creates the session).

### Process for the merge session

1. Open `../3d-parallax-head-hand-tracking-demo/gpu-vision/src/palm-worker.js` and the current `src/palm-worker.js` side by side.
2. Start from the canonical version (has the diagnostics we want to keep).
3. Add: the `useGPUDirect` flag, the `initGPU(device)` parameter, the COPY_DST usage flag on the output buffer, the `gpuLetterboxDirect` function definition, the GPU-direct branch in the per-frame handler that calls `ort.Tensor.fromGpuBuffer`, the "Palm worker: GPU direct path enabled" success log, and the `gpuDirect: useGPUDirect` field in the `ready` postMessage.
4. Reconcile init order: the GPU-direct path requires the ONNX session to exist before `initGPU` is called, so the session creation needs to move earlier. Keep all the existing diagnostic logs around it.
5. Add `enableMemPattern: true` to the session options (Phase 1 added this to the landmark workers but not these — they were skipped because of the bigger merge).
6. Repeat for `face-detection-worker.js`.
7. Test in browser: load `/demos/ball-toss/`, open devtools, look for the `GPU direct path enabled` log and confirm `gpuDirect: true` in the ready message. Run the demo for a minute, watch perf.
8. Commit as one focused commit.

### Also deferred: palm detection model swap

The parallax repo has a different `palm_detection_lite.onnx` than this repo. Full MD5s for identification:

- Parallax repo (`../3d-parallax-head-hand-tracking-demo/gpu-vision/models/palm_detection_lite.onnx`): `a2ffed89a8a4a1ac281e9b25d0ac5427` (3,894,373 bytes)
- This repo (`models/palm_detection_lite.onnx`): `c7442e0d714130ebab375e86fc32fe87`
- Parallax repo also keeps a `palm_detection_lite.onnx.backup` (3,905,734 bytes) next to the active file, which suggests at some point a model swap happened in that repo that was never propagated here. The `.backup` is the one we did NOT take.

I do not know which one is "correct" without testing. **Defer this swap to the same session that does the GPU-direct merge** so both palm-related changes can be verified together. When you do test, run both side by side and compare detection quality, not just whether they load.

## Library divergence (one-time upstream) — completed in Phase 1, partially

The parallax repo's `gpu-vision/src/` is **the better code**. Concrete differences:

- `palm-worker.js`: contains a complete `gpuLetterboxDirect` path (about 50 lines) that keeps letterbox output on the GPU using a shared ONNX/WebGPU device and `Tensor.fromGpuBuffer()`, eliminating the CPU readback. Canonical webgpu-vision is missing this entire function and has no shared-device path.
- `palm-worker.js`: `initGPU()` accepts a shared device. Canonical creates its own device with no sharing path.
- `palm-worker.js`: `useGPUDirect` flag governs the new path.
- `pipeline.js`: keeps `enableMemPattern: true` in the ORT session options. Canonical removed it.
- `model-urls.js`: only difference is `LOCAL_BASE` (`/gpu-vision/models` vs `/models`), which is a path-prefix artifact of where the library was mounted. Canonical's value is correct for this repo.

The only thing canonical has that the parallax copy lacks is some extra `console.error` diagnostics in worker setup, nothing functional.

**Phase 1 outcome: only the safe single-line additions were taken. The bigger improvements were deferred to Phase 1.5 (see above).**

What Phase 1 actually did to `src/`:
- `landmark-worker.js`: added `enableMemPattern: true` to ORT session options
- `face-landmark-worker.js`: added `enableMemPattern: true`
- `face-blendshape-worker.js`: added `enableMemPattern: true`

What Phase 1 left alone (intentionally):
- `pipeline.js`, `face-pipeline.js`: canonical (with snapshot-before-merge WIP) is already a strict superset of the parallax versions; nothing to copy
- `palm-worker.js`, `face-detection-worker.js`: 3-way merge, deferred to Phase 1.5
- `model-urls.js`: paths are already correct for this repo's layout
- `models/palm_detection_lite.onnx`: model swap deferred to Phase 1.5

## Phase 1: Center-of-gravity shift (DONE)

Goal: ship the showcase demo in this repo, untouched-as-possible, importing the library from `../../src/`. No surgery on the demo, no new functionality.

### Steps

1. **Upstream library improvements** (see "Library divergence" above).

2. **Create `demos/ball-toss/`** (folder name is a placeholder; the demo is evolving toward a ball-toss game and will be renamed once it has a real name).
   - Copy `../3d-parallax-head-hand-tracking-demo/index.html` → `demos/ball-toss/index.html`.
   - Copy `../3d-parallax-head-hand-tracking-demo/coi-serviceworker.js` → `demos/ball-toss/coi-serviceworker.js` (the demo needs COOP/COEP for WebGPU).
   - Copy `../3d-parallax-head-hand-tracking-demo/face-worker.js` → `demos/ball-toss/face-worker.js` (the MediaPipe-backend worker; the WebGPU-backend pipeline is loaded from the shared library).
   - Copy `../3d-parallax-head-hand-tracking-demo/hand-worker.js` → `demos/ball-toss/hand-worker.js`.
   - Copy `../3d-parallax-head-hand-tracking-demo/mediapipe-vision.js` → `demos/ball-toss/mediapipe-vision.js`.
   - Edit `demos/ball-toss/index.html`: change the dynamic library imports
     - `await import('./gpu-vision/src/pipeline.js')` → `await import('../../src/pipeline.js')`
     - `await import('./gpu-vision/src/face-pipeline.js')` → `await import('../../src/face-pipeline.js')`
   - Models: the demo loads models via `model-urls.js`, which on `localhost` reads from `/models` and in production reads from `https://models.now.audio`. Both work without changes for this repo's directory layout, so no model copy needed.

3. **Verify Vite serves `demos/ball-toss/`.** This repo already has `vite.config.js` with COOP/COEP headers. Vite serves any HTML file under the root, so `http://localhost:5173/demos/ball-toss/` should Just Work. If it does not, add `demos/ball-toss/index.html` to the rollup `input` map in `vite.config.js`.

4. **Update `README.md`.** Add a "Demos" section near the top:
   ```
   ## Demos
   - `index.html`: hand tracking wireframe (the minimum example)
   - `face.html`: face landmark wireframe
   - `demos/ball-toss/`: full showcase — head-coupled 3D parallax, hand-driven projectiles, MediaPipe vs WebGPU Vision A/B toggle, persisted settings, One Euro filtering
   ```

5. **Smoke test.**
   - `npm run dev`
   - Visit `/` (hand wireframe), `/face.html` (face wireframe), `/demos/ball-toss/` (showcase)
   - All three should work; the showcase should let you toggle between WebGPU and MediaPipe backends and behave identically to how it does in the source repo.

6. **Commit and push.**

### What Phase 1 explicitly does NOT do
- Does not touch `index.html` or `face.html` (still the basic wireframes).
- Does not strip Three.js or refactor the showcase.
- Does not split MediaPipe into "original" vs "workers" backends.
- Does not delete anything.

## Phase 2: Unified one-stop hub (NEXT SESSION) — PARTIALLY DONE

Goal: replace the current basic hand-wireframe `index.html` with a single comparison playground that subsumes both wireframes and gives users one place to A/B everything.

### Already in place (committed in `2b0144d`, tagged `snapshot-before-merge`)

A prior session already moved meaningfully toward this hub, in `index.html` and `src/main.js`. Concrete progress:

- Track toggles for hands and face, persisted to `localStorage`
- Number-of-faces selector with live `FaceTracker` reinit
- Blendshapes enable/show toggles with full dependency cascade (face off forces blendshapes off forces panel off)
- 51-bar live blendshape display panel with hardcoded FACS names
- Visibility-based render pause (loop stops when tab hidden)
- Structured JSON perf logging: `{ engine, active, fps, ms, hands, faces }`

### What still needs to be done

- **Backend dropdown** (the missing centerpiece): WebGPU Vision / MediaPipe (workers) / MediaPipe (original main-thread). The current hub WIP only runs WebGPU Vision. Wiring up the MediaPipe paths is the main net-new work.
- **`face.html` deletion**, after the hub is verified covering all face-tracking use cases (including the blendshape panel layout that was originally on `face.html`).
- **Hand-side parity check**: confirm the hub draws hand wireframes the same way the existing minimal `index.html` did before it was overwritten by hub WIP. If anything regressed, port it from git history.
- **README update** to describe the hub as the primary entry point.

### Hub spec (the one-stop shop)

A single page where the user picks what to track and which backend to run:

- **Track toggles**: head on/off, hands on/off, hand count (1 or 2)
- **Backend dropdown**: WebGPU Vision / MediaPipe (workers) / MediaPipe (original main-thread)
  - "MediaPipe (workers)" is the existing worker-based path from the parallax demo (`face-worker.js`, `hand-worker.js`)
  - "MediaPipe (original)" is a new code path that runs MediaPipe on the main thread, for full A/B comparison. This is the only piece of new implementation in Phase 2.
- **Future toggle (Phase 3)**: face blendshapes display on/off
- **Visualization**: video feed + 2D wireframe overlays. **No Three.js, no 3D scene, no projectiles.** Pure tracking visualization.

### Steps (revised in light of existing WIP)

The original plan was "start from a copy of `demos/ball-toss/index.html` and strip Three.js." That is no longer the right starting point, because the current hub WIP in `index.html` is already a stripped, 2D-only structure with most of the toggle UI in place. **Build on top of it instead.**

1. **Start from current `index.html` + `src/main.js`** (the snapshot-before-merge state). Do not re-derive from the parallax demo.

2. **Add the backend dropdown.** Insert a `<select id="backend-select">` with options `WebGPU Vision`, `MediaPipe (workers)`, `MediaPipe (original)`. Persist selection to `localStorage`. Refactor `src/main.js` so the per-frame work goes through one of three init/teardown pairs depending on selection. The "MediaPipe (workers)" path can be lifted from `demos/ball-toss/face-worker.js` and `hand-worker.js`. The "MediaPipe (original)" path is new code: import MediaPipe Tasks Vision and run on the main thread.

3. **Verify hand wireframe rendering** still works exactly as the pre-WIP `index.html` did. The hub WIP added face/blendshape support but I have not yet verified it left the hand-tracking display intact. Diff against `7dbb45e` (last commit before the WIP) and confirm hand drawing was preserved.

4. **Verify face wireframe + blendshape panel** matches what `face.html` provided.

5. **Delete `face.html`** once steps 3 and 4 pass.

6. **README update** describing the hub as the primary entry point.

### Steps (original plan, now obsolete — kept for reference)

~~1. **Start from a copy.** Copy `demos/ball-toss/index.html` → working file for the new `index.html`.~~

~~2. **Strip Three.js and the 3D scene.** Remove:~~
   - The `three` import map and `import * as THREE`
   - The scene/camera/renderer/lighting setup (~lines 162 onward in the source)
   - The background grid, cube grid, projectile pool, hand-skeleton 3D meshes
   - The parallax controls (`#parallax-select`, `#parallax-amount`, `#parallax-depth`)
   - The animation loop's 3D rendering side
   Keep:
   - The video element + 2D overlay canvas
   - The backend dropdown and its `initWebGPUBackend` / `initMediaPipeBackend` paths
   - The persisted-settings logic
   - The One Euro filter wiring (still useful for smoothing 2D landmarks)
   - The HUD (FPS, latency, backend display)

3. **Add 2D wireframe rendering** to the overlay canvas. The basic wireframes in current `index.html` and `face.html` already do this — port that drawing code into the new hub.

4. **Add the head/hands track toggles.** Wire them so the appropriate worker(s) are spun up or torn down dynamically.

5. **Wire up "MediaPipe (original)" backend.** Add a new init/teardown pair that runs MediaPipe Tasks Vision on the main thread (no worker). This is the only net-new code in Phase 2.

6. **Verify against the existing wireframes.** With backend = "WebGPU Vision" and head off, hands on, the new hub should look and behave identically to the current `index.html`. With head on, hands off, it should match `face.html`.

7. **Replace `index.html` with the new hub.** Delete `face.html` only after the hub is verified working in all combinations.

8. **README update.** Replace the "Demos" section with the hub description plus the showcase link.

### Phase 2 risks / things to watch
- The source `index.html` mixes 2D pose state with 3D mesh updates. The strip pass needs to keep the state-update side and remove only the rendering side. Easy to accidentally drop a state update.
- "MediaPipe (original)" needs to load the same WASM and models the worker version uses. Make sure the WASM URLs match so we are not measuring CDN cold-start differences in the comparison.
- COI service worker is still required for WebGPU; keep `coi-serviceworker.js` registered in the new hub.

## Phase 3: Polish and longer-term

These are tracked here so they do not get lost; none are blocking a release.

### Investigate: why hand pipeline gets ~10x speedup but face only ~2-3x

Live ball-toss bench on M1 Max, post-Phase-1.5, shows:

| | MediaPipe | WebGPU Vision | Speedup |
|---|---|---|---|
| Hand (2 hands) | 40-48ms mean, 56-60ms p95 | 0.7-6ms mean, 11-16ms p95 | ~8-10x |
| Face Detector (BlazeFace) | 18-21ms mean, 27-30ms p95 | 7-8ms mean, 11ms p95 | ~2.5-3x |
| Face Landmark (FaceMesh) | 27-31ms mean, 37-38ms p95 | 13.5ms mean, 20ms p95 | ~2x |

The hand delta is much bigger than the face delta. **Why** is worth understanding before we publish numbers, because the answer shapes the pitch.

Hypotheses to test (in rough order of likely contribution):

1. **Two-hand parallelism on our side, sequential cascade on MediaPipe's side.** Our `HandTracker` runs the two landmark workers under `Promise.all` so they execute on separate worker threads in true parallel. MediaPipe's `HandLandmarker` runs the per-hand landmark cascade sequentially within a single graph. So at 2 hands, our advantage is roughly (cascade-overhead-removal) x (2x parallelism). Face has no equivalent — only one face, no parallelism axis. To test: rerun both backends at `numHands=1`. If WGPU hand speedup at 1 hand drops to ~3-5x, parallelism is the dominant factor.

2. **Cascade depth = readback count.** MediaPipe pays a `glReadPixels` between every model in a cascade. Hand cascade for 2 hands = 1 palm detect + 2 landmark = 3 model calls = 3 readbacks. Face landmark cascade = 1 detect + 1 landmark = 2 readbacks. More readbacks per frame = more pain we save by removing them.

3. **Model size & inference time.** Hand landmark is a smaller model than face landmark (3.9 MB vs 4.8 MB, but face mesh runs at 256x256 which is more pixels). Smaller/cheaper inference = readback is a bigger fraction of total time = bigger relative win when readback is removed. Face mesh inference itself is heavier so removing readback is a smaller fraction of the total.

4. **PReLU decomposition tax (face-only headwind).** Our face landmark model has 69 PReLU ops decomposed into Relu+Neg+Mul+Add for WebGPU compatibility. MediaPipe's TFLite has PReLU as a native op. So the face landmark inference has a small structural disadvantage on our side that the hand landmark does not. See [PRELU_DECOMPOSITION.md](PRELU_DECOMPOSITION.md).

How to investigate: rerun the ball-toss bench with `numHands=1` (eliminates parallelism), then with the hand model running through a single landmark worker (eliminates per-hand parallelism while keeping the same cascade), and compare deltas. The numbers should tell us which hypothesis carries the most weight. Worth 30 minutes of experiment time before any public benchmark publication.



### Hand identity tracking: stable slot assignment across exits/entries (NEXT)

**The problem:** When one hand leaves frame and returns, it can get assigned to the wrong slot, causing the visual wireframes to swap sides. When hands clap, both slots can lock onto the same physical hand. MediaPipe has the exact same issue -- this is fundamental to how HandLandmarker returns results (no persistent identity, just an ordered array that reshuffles when hands enter/leave).

**What we tried (2026-04-15) and why it didn't work:**
- Position-based slot assignment (left side -> slot 0, right side -> slot 1): fails because camera is mirrored, and a detection at the frame edge of the leaving hand steals the slot
- Handedness swap after inference: causes visible one-frame flip when hands re-enter
- Edge rejection (reject detections near frame borders): shrinks playable area, still doesn't prevent the swap
- Cooldown on dropped slots: hack, and the detection still comes back after cooldown expires
- Spatial proximity to last-known position: closest to correct but the edge-of-frame phantom detections are spatially close to the wrong slot's last position

**The real fix (updated 2026-04-15 per Robert):** WebGPU Vision wraps around the model and fixes handedness BEFORE the end-user interacts with it. Library-level feature, not demo-level workaround. Palm centroid-based identity:

1. Compute a very low-cost centroid from 4 of the most stable palm points (probably wrist + MCP joints -- less jitter than fingertips)
2. Each slot maintains its own centroid history -- OUR identity, independent of model labels
3. When hand count changes (2->1 or 1->2), compare incoming centroids against slot centroids. Assign by nearest match. Handle the flip on THAT exact frame.
4. Model's handedness is a hint only, used for initial assignment when both hands first appear. After that, centroids own identity.
5. MediaPipe's pipeline stays broken -- that's their problem. WebGPU Vision returns stable hand identities as a selling point.

Fix goes in `pipeline.js`, not the demo.

**What's in the pipeline now (from this session, should be cleaned up):**
- Dedup detection (background palm detection when same handedness + overlap for 3 frames) -- this is still useful and should stay
- `lastRect` on slots for spatial assignment -- partially implemented, can be simplified
- Handedness swap was removed -- keep it removed, it caused visual flips
- Edge rejection was reverted -- correct, don't shrink play area

### Known issue: hand identity can swap when hands fully overlap

WebGPU Vision hand tracking (2026-04-17) is at parity with or better than MediaPipe's demo behavior in most scenarios, but when two hands overlap heavily (palms together / one hand directly behind the other), one slot occasionally "swallows" the other -- both slots briefly track the same physical hand before the dedup system recovers.

Root cause: our landmark model is given a per-slot ROI crop. When one slot's 2x-expanded ROI covers both hands, the model returns landmarks for whichever hand is more prominent, so both slots converge. The pipeline now handles this with:
- **Centroid-based identity tracking** (palm of 4 MCP+wrist points) -- hands that just *approach* each other don't swap
- **Duplicate detection** (tunable threshold, default 25px, slider in drawer) -- catches convergence and drops the losing slot so palm detection can re-establish
- **Palm re-anchor during overlap** (<80px centroid distance) -- fires palm detection every frame while hands overlap and re-anchors each slot's ROI to the matching palm-detected hand

Still imperfect for full overlap cases. MediaPipe's pipeline has additional tracking-state logic (likely cross-frame assignment with Hungarian-style cost matrices plus internal smoothing) that we haven't replicated. Our landmark model is the same as theirs (Google's hand_landmarker) so the gap is purely in the orchestration layer. Revisit when centroid-tracking proves insufficient in production -- for now, document and ship.

### Known issue: intermittent startup stall (20-30 seconds)

Observed 2026-04-15. Demo loads, all workers report ready, camera goes live, face detection fires once and assigns slot 0, palm detection fires once (5 detections, assigns one hand), then **everything stalls for 20-30 seconds**. No FRAME RATE logs, no BENCH logs, tracking stuck on `_,_`. Eventually recovers with a **22x oversampling burst** (673 hand samples, 668 face samples in one report) then settles to normal 30fps.

The 22x oversampling is the smoking gun: the video frame callback was accumulating frames without processing them, then fired them all at once. Classic signature of either (a) tab being throttled/backgrounded, (b) an async init step blocking frame processing, or (c) a worker getting stuck on a promise that eventually resolves.

Not reliably reproducible yet. Worth investigating next time it happens: check `document.visibilityState` at startup, whether any worker's first `postMessage` is hanging, and whether `requestVideoFrameCallback` has a warmup issue. The face worker's very first detection took an abnormally long time to complete (face-lm p95 was 106.7ms max in that recovery burst, vs typical 14-16ms), which suggests the stall is on the face pipeline side.

### Known issue: tracking never engages on cold start (tab-away-and-back fixes it)

Demo loads and renders the 3D scene, but nothing else happens -- no camera self-view, no hand tracking, no head tracking. Scene is completely inert. **Tabbing away to another browser tab and tabbing back fixes it every time** -- the `visibilitychange` handler's re-kick of the rAF + rVFC chains is sufficient to unstick it. Points to a race condition in the initial startup sequence. See [start_failure_log.md](start_failure_log.md) for full analysis.

### Known issue: face detection tracking preview not cleaned up on toggle off

When the face detection model is toggled off in the ball-toss demo, the small tracking preview window in the corner (showing the detected face region) is not cleaned up -- it stays on screen even after face tracking stops. Demo-level issue in `demos/ball-toss/index.html`. Fix: when face tracking is disabled, clear the preview canvas and hide the overlay element.

### Mobile / phone support: WebGPU Vision not loading

Current symptom: the demo loads on desktop Chrome but **fails to load on phone**. Root cause not yet diagnosed. Investigation steps:
- Connect phone via remote devtools (Chrome on Android: `chrome://inspect`; Safari on iOS: develop menu)
- Check whether the issue is at the WebGPU adapter level (`navigator.gpu.requestAdapter()` returning null), at the model fetch level (CORS, MIME, size), at the ONNX Runtime initialization level, or at the worker spawn level
- Test on multiple mobile browsers: Chrome Android, Safari iOS, Edge Android
- Note: iOS Safari WebGPU support is gated behind a feature flag in older versions; verify the test device's iOS version
- If WebGPU is unavailable on the test phone, the demo should at least fail gracefully and surface a clear message instead of just hanging

### Asset hosting and local persistence

Three intertwined questions about model file delivery:

1. **Ideal download path / CDN strategy.** Currently models are served from `https://models.now.audio` in production via [model-urls.js](src/model-urls.js). Open question: is that the best home long-term? Alternatives: GitHub Releases (free, slower, has bandwidth limits), HuggingFace Hub (purpose-built for model hosting, free, fast), Cloudflare R2 (cheap, fast, no egress fees), keeping `models.now.audio`. Decide before any public release push.

2. **Local persistence (caching).** Currently every page load re-downloads ~15MB of ONNX models. We should cache them in IndexedDB or Cache API on first load and read from local storage on subsequent loads. Even better: use the Cache API with the service worker (`coi-serviceworker.js` is already in place for COOP/COEP) so the browser handles staleness. Saves bandwidth and dramatically improves return-visit cold start.

3. **Toggle to disable local persistence.** For development and testing we need a setting (URL param or localStorage flag) that **forces a fresh download bypassing the cache**, so we can test loading from different CDN paths and verify that production load behavior matches dev. Default: cache on. Override: `?nocache=1` or similar.

These three should probably be tackled together — once we have a caching layer, the CDN choice and the toggle naturally fall out.

### Demo polish
- **Proper hand Z-axis depth:** Currently the hand wireframe Z position is a rough offset from camera (`camera.z - 6 + lm.z * -3`). The hand landmark model outputs a real Z coordinate per joint, but we're only using it as a small perturbation. The fix: map the landmark Z values to actual world-space depth so that (a) the hand moves along the Z axis as you push it toward/away from the camera, (b) the wireframe size stays fixed in world units so it gets smaller with distance (perspective projection handles this naturally once Z is correct), and (c) the held ball and throw velocity incorporate real depth. This requires calibrating the landmark model's Z output range to the Three.js scene's coordinate system. The parallax compensation in `landmarkToScene` (the `pFrac = handZ / camera.z` factor) already handles the camera offset correctly — just the Z mapping needs work.
- Rename `demos/ball-toss/` to whatever the game ends up being called.
- Build out the actual ball-toss game mechanics (currently the demo throws projectiles when you pinch; the full game is TBD).
- Add a "blendshape display" toggle to the hub (face blendshape worker already exists in `src/face-blendshape-worker.js`; just needs UI).

### Model provenance and hosting
Open question: did we convert the ONNX models ourselves from MediaPipe weights, or did we download pre-converted ones? This matters for the open-source story. Investigation steps:
- Check this repo and the parallax repo for any conversion scripts (`*.py`, `convert*`, `onnx*` outside `node_modules`).
- Check git history of `models/` for the original commit message.
- If we converted: document the conversion process in `MODELS.md` for reproducibility.
- If we downloaded: confirm the source license permits redistribution and credit it in `models/LICENSE.md` (which already exists; verify it covers all five files).

Hosting is currently `https://models.now.audio` in production via [model-urls.js](src/model-urls.js). That is fine for now. Longer term, decide whether to host on GitHub Releases, HuggingFace, or keep `models.now.audio`.

### Library structure
- The PReLU decomposition documented in `PRELU_DECOMPOSITION.md` is core to the face landmark model running on WebGPU. Worth a callout in the README for credibility.
- The benchmark in `benchmark/` should be wired into the hub eventually so users can run their own A/B and see numbers, not just animation.

### Documentation
- `ARCHITECTURE.md` is good. Add a "Demos" section pointing to the hub and the showcase.
- The "Hold my bear" line in the README is gold. Keep it.

## Phase 4: Drop ONNX Runtime entirely — custom WGSL inference engine

**Status: ALL FOUR MODELS VERIFIED (2026-04-12).** The custom WGSL inference engine is built and produces output matching ONNX Runtime within 0.0005 for all four models. Not yet wired into the demo — still running from test harnesses.

### Why this just became urgent (2026-04-12)

ONNX Runtime's WebGPU backend **does not work on iOS Safari**. At all. Not our code, not our hosting, not our headers — Microsoft's ORT simply doesn't support iOS WebGPU ([issue #22776](https://github.com/microsoft/onnxruntime/issues/22776)). Even on Safari 26 desktop it has severe CPU/memory issues ([issue #26827](https://github.com/microsoft/onnxruntime/issues/26827)). This means the demo Robert built — the thing that proves WebGPU Vision works — cannot run on any iPhone. MediaPipe is NOT a fallback (see memory: this repo demos WebGPU Vision, period). So either we wait for Microsoft to fix ORT on Safari (timeline unknown), or we drop ORT entirely and run the models ourselves.

### What dropping ORT means

We have the ONNX model files. They contain the weights AND the computation graph. A neural network inference is:

1. Read the graph: "input → Conv2D → Relu → Conv2D → Relu → ... → output"
2. For each op, dispatch a WebGPU compute shader that runs the math using the stored weights
3. Feed the output buffer to the next op

Our four models use maybe **15-20 unique operation types** (Conv2D, DepthwiseConv2D, MatMul, Relu, PReLU, Add, Mul, Reshape, Transpose, Sigmoid, Pad, Resize, Concat, Softmax, BatchNorm). Each op is a WGSL compute shader, ~50-100 lines each. The "inference engine" is a loop that walks the graph and dispatches shaders in order.

**Estimated scope:**
- ~15-20 WGSL compute shaders for the ops our models use
- A graph parser (read ONNX protobuf or pre-convert to JSON at build time)
- A graph walker that allocates GPU buffers and dispatches shaders in order
- ~2000-3000 lines of JS/WGSL total
- Download: ~50KB instead of 23MB
- Zero WASM, zero threads, zero SharedArrayBuffer, zero crossOriginIsolated requirement
- Works everywhere WebGPU works — including iOS Safari

**What we lose:** ONNX Runtime's graph optimizer, operator fusion, memory planner, and the ability to run arbitrary ONNX models. We don't need any of that — we run exactly four small, known, fixed models.

**What we gain:** iOS support, ~450x smaller download, no WASM dependency, no COOP/COEP header requirement, no service worker hacks, works on GitHub Pages natively, total control over the inference pipeline.

### Confirmed: WebGPU compute works on iOS Safari (2026-04-12)

Deployed a bare-bones sanity check (`webgpu-test.html`) to Cloudflare Pages. On an iPhone running iOS 26.4.1, Safari:
- `navigator.gpu`: true
- `requestAdapter()`: success
- `requestDevice()`: success
- Compute shader (multiply array by 2): correct output
- **Result: SUCCESS**

WebGPU compute is fully functional on iOS Safari. The ONLY thing blocking our demo was ONNX Runtime's WASM backend. Pure WGSL compute shaders run fine. This confirms Phase 4 is not speculative — the target platform works.

### No more COOP/COEP headers

The entire reason we needed `Cross-Origin-Embedder-Policy` and `Cross-Origin-Opener-Policy` headers was `SharedArrayBuffer`, which ONNX Runtime's multi-threaded WASM requires. Pure WebGPU compute shaders don't use SharedArrayBuffer. No WASM threads, no special headers needed.

This means:
- **GitHub Pages works natively** (no service worker shim)
- **Any static host works** (S3, Netlify, Vercel, Cloudflare, anything)
- **No `crossOriginIsolated` requirement**
- The `coi-serviceworker.js` hack can be deleted entirely
- Just HTML + JS + WGSL. Open the page, it runs.

### Fused mega-shaders: why we'll be FASTER than ORT

ORT runs each neural network operation as a separate GPU shader dispatch. Conv2D, then BatchNorm, then Relu — three dispatches, three intermediate buffers, three round-trips through the GPU command queue. It has to do this because it's general-purpose: it doesn't know at compile time which ops will be adjacent.

We know exactly which ops are adjacent. Our models are fixed. So we fuse entire blocks:

| ORT approach (6 dispatches) | Fused approach (1 dispatch) |
|---|---|
| Conv2D → buffer → BatchNorm → buffer → Relu → buffer → Conv2D → buffer → BatchNorm → buffer → Relu | Conv+BN+Relu+Conv+BN+Relu as ONE shader |

Specific fusions available in our models:
- **Pre-bake BatchNorm into Conv2D weights at build time.** BN is just `(x - mean) / sqrt(var) * scale + bias` — fold scale/bias into the conv weights offline. Zero runtime cost, one fewer op.
- **Fuse Conv+Relu** — clamp output at the end of the convolution loop. Trivial.
- **Fuse entire residual blocks** — Conv+BN+Relu+Conv+BN+Add+Relu as one monolithic shader. One dispatch instead of seven.
- **Inline PReLU** — we decomposed PReLU into 4 ops (`Relu(x) + slope * -Relu(-x)`) for ORT compatibility. A custom shader is just `x > 0 ? x : slope * x`. One instruction instead of four dispatches.

Result: maybe **5-8 fused mega-shaders per model** instead of 60-80 individual dispatches. Fewer dispatches = less GPU idle time = potentially faster than ORT even on desktop.

### Verification: zero secret sauce

ONNX is an open format. Every operation, weight, shape, and connection in our models is fully readable:
```python
import onnx
model = onnx.load("palm_detection_lite.onnx")
for node in model.graph.node:
    print(node.op_type, [i for i in node.input], [o for o in node.output])
```

We can verify correctness by running ORT and our custom engine side-by-side, comparing output tensors at every stage. `GPUCommandEncoder` timestamps give nanosecond-precision per-dispatch benchmarks. Every step is measurable, every result is verifiable.

### Incremental build plan

Each step produces a working, testable artifact. We never go more than a day without something we can benchmark against ORT.

**Step 1: Model surgery ✅ DONE (2026-04-12)**
All 4 models extracted: graph JSON + flat weight binary for each. BatchNorm already pre-baked by ONNX export. Face landmark PReLU decomposition (6 ops) fused back to native PReLU (1 op) during graph preprocessing.

**Step 2: Palm detector ✅ DONE**
5 WGSL shaders (conv2d, maxpool, resize, add, pad_channels) + 2 more (gemm, global_avg_pool). Palm runs end-to-end, output matches ORT within 0.000211. 1.8x faster than ORT-GPU on palm.

**Step 3: Hand landmark ✅ DONE**
All 4 outputs match ORT (landmarks, hand flag, handedness, world landmarks). Generic ModelRunner handles the full MobileNetV2 architecture. Added ReLU6, Gemm+Sigmoid, GlobalAvgPool support.

**Step 4: Face detector + face landmark ✅ DONE (blendshapes still TODO)**
Both models verified. Face detector needed standalone Relu fix. Face landmark needed standalone PReLU fix + PReLU re-fusion from decomposed form. Blendshape model (face_blendshapes.onnx) not yet extracted.

**Step 5: Optimize ✅ DONE (2026-04-13)**

Five optimization passes in one session, each building on the last:

1. **GPU PReLU + uniform buffer pool** — moved 34 standalone PReLU ops from CPU readback to GPU dispatch (add.wgsl mode 3). Pooled uniform buffers across dispatches. Face landmark: 53.61ms -> 14.68ms.

2. **Level 1 kernel fusion** — fused DW Conv + 1x1 Conv + Add + Activation into single `fused_block.wgsl` dispatch. DW output stays in registers, never hits memory. Smart gating: skip fusion when 1x1 narrows channels (redundant DW recompute) or when `dwInCh * kArea > 1024` (too much work per thread for large kernels). Handles asymmetric padding for stride-2 blocks and optional DW activation (ReLU6/ReLU between DW and 1x1).

3. **GPU transpose** — NCHW->NHWC transpose moved from CPU readback to `transpose_nhwc.wgsl` compute shader. Output head assembly (Concat) now stays entirely on GPU using `copyBufferToBuffer`.

4. **Pre-compiled command replay** — `compile()` walks the graph once, capturing all dispatches, bind groups, and buffer copies into a flat steps array. `runCompiled()` replays with zero graph walking, zero allocation, zero bind group creation.

5. **Pre-allocated readback** — staging buffers for output readback created once during `compile()`, reused every frame. Output copies baked into the same command encoder as dispatches. Parallel `mapAsync` on all outputs.

**Benchmark results (headless Chrome, M1 Max, compiled path, 50 iterations):**

| Model | Before | After | ORT WASM | vs ORT | Speedup |
|---|---|---|---|---|---|
| Palm Detector | 18.61ms | **13.32ms** | 27.47ms | **2.06x faster** | 1.40x |
| Hand Landmark | 12.67ms | **6.14ms** | 17.31ms | **2.82x faster** | 2.06x |
| Face Detector | 12.70ms | **3.30ms** | 3.11ms | ~parity | 3.85x |
| Face Landmark | 53.61ms | **8.49ms** | 13.41ms | **1.58x faster** | 6.31x |

Every model beats or matches ORT WASM. Three out of four are 1.5-2.8x faster. Face landmark improved 6.3x from the naive baseline.

**vs MediaPipe (Google's sealed WASM/WebGL SDK):**
MediaPipe's browser SDK runs at roughly 30-40ms combined for hand + face in the ball-toss demo, bottlenecked by synchronous `glReadPixels` readbacks (8-22ms each). Our compiled WGSL engine runs the same models at:
- Palm + face detection in parallel: **max(13.32, 3.30) = 13.32ms**
- Then hand landmark per detected hand: **6.14ms**
- Total per-frame inference: **~19ms = 52 fps** (vs MediaPipe's ~25-33 fps)
- And we haven't even wired it in yet -- once running in workers with parallel hand slots, effective latency drops further.

**Step 6: Wire into demo ✅ DONE (2026-04-13)**

All four inference workers replaced with WGSL engine (palm-worker-wgsl.js, landmark-worker-wgsl.js, face-detection-worker-wgsl.js, face-landmark-worker-wgsl.js). Each worker has its own GPU device for true parallel execution. GPU sigmoid added (add.wgsl mode 4) to fix stale handFlag/handedness in compiled path. MediaPipe backend now fully torn down on switch to free GPU resources.

**Step 6b: Shader optimization + architecture tuning ✅ DONE (2026-04-14)**

Shader-level compute optimizations gave the biggest wins:
- Workgroup shared memory for depthwise conv: 2x hand speedup (6.4ms -> 3.2ms headless)
- Vec4 dot product for 1x1 pointwise: significant contribution to all models
- Tried and rejected: inverted residual fusion (9x redundant compute), weight tiling (barrier cost > cache savings), bitmap cloning (sequential worse than parallel)

Architecture: separate workers (5 GPU devices) with optimized shaders outperformed unified worker on all metrics. M1 GPU driver handles multi-device efficiently.

**Live demo benchmarks (final, M1 Max, Chrome, 480x360, 1.0x sampling):**

| | WGSL (live) | ORT-WebGPU (old) | MediaPipe | vs ORT | vs MediaPipe |
|---|---|---|---|---|---|
| Hand | **9.5ms** | 8.2ms | 29.3ms | 16% slower | **3.1x faster** |
| Face LM | **13.2ms** | 13.0ms | 25.1ms | **parity** | **1.9x faster** |

**Headless benchmarks (the compute ceiling, 2026-04-21):**

| Model | **WGSL** | **ORT WebGPU** | **ORT WASM** | vs ORT-GPU | vs ORT-WASM |
|---|---|---|---|---|---|
| Palm | **9.34ms** | 24.40ms | 28.90ms | **2.6x** | **3.1x** |
| Hand | **3.03ms** | 6.89ms | 18.02ms | **2.3x** | **5.9x** |
| Face det | **3.13ms** | 3.33ms | 3.02ms | **1.1x** | 0.97x |
| Face LM | **5.88ms** | 8.27ms | 13.86ms | **1.4x** | **2.4x** |

Run `node engine/bench-all.mjs` to reproduce (50 iterations, 20 warmup, headless Chrome).

**UPDATE (2026-04-15): ORT BEATEN.** Four optimizations in one session:
1. Cached warp texture + bind group (eliminated per-frame GPU allocations)
2. Pre-allocated readback arrays (eliminated per-frame JS heap allocations)
3. VideoFrame zero-copy transfer (0.02ms vs 0.5ms per createImageBitmap call)
4. Merged warp + inference into single GPU submit (one submit per worker per frame)

**UPDATE (2026-04-20): Shader compute round 2.** Four more optimizations:
1. GEMM parallel reduction + vec4 (64-thread shared memory reduction for large matrices)
2. Conv2D 1x1 double-unrolled vec4 + adaptive oc_tile (oc_tile=4 for small channels, 1 for large)
3. Conv2D unrolled 2x2 general path (52% faster face landmark strided downsamples)
4. Fused block tiled variant (separate pipeline with shared memory for spatial <= 8x8)

**Current live benchmarks (M1 Max, Chrome, 480x360, 2 hands + face):**

| | WGSL | ORT-WebGPU | MediaPipe | vs ORT | vs MediaPipe |
|---|---|---|---|---|---|
| Hand | **7.9ms** | 8.2ms | 29.3ms | **1.04x** | **3.7x** |
| Face LM | **12.4ms** | 13.0ms | 25.1ms | **1.05x** | **2.0x** |

Runs on iOS Safari (pure WebGPU, no ONNX Runtime, no WASM). Confirmed working on iPhone.

**What's left for Step 7: Ship + Demo Polish**
- **Hand identity tracking** (DONE 2026-04-17): pipeline now uses palm-centroid identity tracking, duplicate detection with tunable slider in the drawer, and palm re-anchor while hands overlap. At parity with or better than MediaPipe in most cases. Full overlap still imperfect -- see "hand identity can swap" known issue.
- **SpellARia Motion Signature Recorder (CRITICAL PATH)**: build the data-collection tool NOW, start collecting a labeled dataset of reversals vs terminations. This is the path to the prediction classifier that eliminates perceived latency. Data collection is the long pole -- wall-clock time spent physically performing motions. See "CRITICAL PATH: SpellARia Motion Signature Recorder" section below.
- **RTMPose hand-only evaluation** (DONE 2026-04-20): RTMPose ruled out (too big, too slow). Discovered `hand_landmark_10mb` (10.1MB, 2,643,047 params, same MediaPipe family, vastly superior pinch tracking). See "RTMPose hand-only evaluation" section below.
- **Port hand_landmark_10mb to WGSL engine** (NEXT UP, HIGH PRIORITY): same architecture as hand_landmark_4mb, just wider layers. ONNX file exists at `models/hand_landmark_sparse_Nx3x224x224.onnx` (10.1MB, 2,643,047 params). Needs `onnx_to_json.py` conversion script (doesn't exist yet -- `onnx` Python package is installed). Compile through ModelRunner. Should need zero new ops. Estimated ~5-8ms/hand vs current ~3-4ms/hand. Batch dimension enables both-hands-in-one-call. Better pinch/thumb tracking + potentially better z signal for orientation work. See "Next: port hand_landmark_10mb to WGSL engine" section below.
- **Z-axis depth estimation** (IN PROGRESS 2026-04-24): World landmarks wired through pipeline. Discovered world landmarks are hand-relative (rotate with the hand), NOT camera-relative -- useless for orientation but give rotation-invariant bone lengths in meters (~6.3cm palm width). Current approach: max(screenW/worldW, screenH/worldH) picks the bone most perpendicular to camera as distance proxy. Ball + hand-translation Z modes added to demo. Still tuning rotation invariance. See HAND-LANDMARK-OUTPUTS.md.
- **Hand orientation axes** (IN PROGRESS 2026-04-24): RGB arrow visualization of palm coordinate frame. World landmarks can't solve this (hand-relative frame). Normalized landmarks need better z handling. Unsolved -- see HAND-LANDMARK-OUTPUTS.md for full analysis of failed approaches.
- Hand parallax compensation using the derived z-depth.
- **Output adapter layer + OSC pipeline**: pluggable outputs (OSC, MIDI, WebSocket, BroadcastChannel) so hand/gesture events can drive other software and devices. See "Future: output adapters + OSC pipeline" section below.
- **PFLD face tracking alternative**: Apache 2.0, ~98 2D keypoints, lightweight and fast. Second face backend for demos/users who don't need MediaPipe's full 478-point mesh + blendshapes. Not an upgrade path (MediaPipe face is already excellent); purely a library-flexibility demo. See "Face tracking alternative: PFLD" section below.
- Delete `vendor/onnxruntime-web/` (23MB -- still needed for blendshape worker)
- Extract face blendshape model to WGSL (then vendor/ can be fully deleted)
- Push to GitHub Pages

**New engine files added this session:**
```
add.wgsl             — Now includes mode 4 (sigmoid) -- zero CPU ops remaining
palm-worker-wgsl.js  — Palm detection on compiled WGSL engine
landmark-worker-wgsl.js — Hand landmark on compiled WGSL engine  
face-detection-worker-wgsl.js — Face detection on compiled WGSL engine
face-landmark-worker-wgsl.js — Face landmark on compiled WGSL engine
```

**Current engine files (in `engine/` directory):**
```
conv2d.wgsl          — Conv2D + PReLU/ReLU6/ReLU + residual Add (the workhorse)
fused_block.wgsl     — Fused DW Conv -> [Act] -> 1x1 Conv -> [Residual] -> Activation
maxpool.wgsl         — MaxPool 2x2 + channel padding
resize.wgsl          — Bilinear 2x upsample (FPN heads)
transpose_nhwc.wgsl  — NCHW -> NHWC transpose for output heads
add.wgsl             — Element-wise ops: add, relu, add+relu, prelu
pad_channels.wgsl    — Channel zero-padding for residual connections
gemm.wgsl            — Matrix multiply + optional sigmoid (FC layers)
global_avg_pool.wgsl — Global average pooling
model-runner.js      — Graph walker + compile/runCompiled for zero-overhead replay
session-timer.mjs    — Session time tracking (reads ~/.session-timer/)
test.html            — Palm detector test (ModelRunner + compiled)
test-hand.html       — Hand landmark test (ModelRunner + compiled)
test-face-det.html   — Face detector test (ModelRunner + compiled)
test-face-lm.html    — Face landmark test (ModelRunner + compiled)
```

### Future: model upgrades once the engine exists

The custom WGSL engine doesn't care which model's weights it runs. The immediate upgrade is **hand_landmark_10mb** (see "RTMPose hand-only evaluation" section below) -- same architecture as our current hand_landmark_4mb, just wider, with vastly better pinch tracking.

**RTMPose evaluated and ruled out (2026-04-20):** RTMPose-m hand is 13x larger (53MB, 13.59M params) and 5.5x slower than our current pipeline. Only one hand variant exists (medium, alpha). fp16 gives zero inference speedup on Apple Silicon. Nobody has shipped it in-browser. The accuracy advantage on prayer/pinch hands does not justify the cost when hand_landmark_10mb achieves similar gains at 10.1MB. See full evaluation below.

Other options surveyed (2026-04-12):
- **OpenPose**: best accuracy but non-commercial license. Dead end for a product.
- **YOLOv8/v11 Pose**: good accuracy, fast, but AGPL license (or paid commercial). Risky.
- **MMPose (general)**: research toolkit, many models, Apache 2.0. RTMPose is their production-grade export -- evaluated and too heavy for browser.

The engine's flexibility remains: any ONNX model that uses our supported ops can be compiled to WGSL. But for hand tracking, the MediaPipe model family covers the accuracy-size spectrum we need: hand_landmark_4mb (current, 1M params) and hand_landmark_10mb (upgrade, 2.6M params).

### RTMPose hand-only evaluation: COMPLETED (2026-04-20)

**Result: RTMPose is NOT viable for browser real-time hand tracking.**

Session 2026-04-20 built a full A/B test page (`engine/test-rtmpose.html`) comparing our WGSL pipeline, RTMPose fp32, RTMPose fp16, and the MediaPipe sparse landmark model. Key findings:

**RTMPose benchmarks (via ORT WebGPU EP on M1 Mac):**

| Model | Landmark time | Model size | Accuracy |
|---|---|---|---|
| Our WGSL pipeline (hand_landmark_4mb) | ~10.5ms full pipeline | 4.1MB | Good, struggles with prayer hands |
| RTMPose-m fp32 | ~57ms landmark only | 53MB | Better on prayer hands |
| RTMPose-m fp16 | ~55ms landmark only | 26MB | Same as fp32 (no speedup on M1) |
| **hand_landmark_10mb (batched)** | **~11ms both hands** | **10.1MB** | **Vastly superior pinch tracking** |

**Why RTMPose failed:**
- Only one hand model variant exists (medium, 13.59M params, 2.6 GFLOPs). No tiny/small hand versions. The body pose has t/s/m/l but hand is alpha and medium-only.
- 13x more parameters than MediaPipe's hand landmark (~1M params). The 5.5x speed gap is expected.
- fp16 conversion halved download size (53MB -> 26MB) but gave zero inference speedup on M1 -- GPU runs the same computation, just with smaller weights in memory.
- Loading the 53MB model caused GPU contention that degraded our WGSL pipeline's concurrent inference.
- Nobody has shipped RTMPose hands in-browser. Zero working browser demos found anywhere.
- ORT WebGPU dispatch overhead: hundreds of individual GPU dispatches at ~24-36us each adds 10-20ms of overhead on top of the model's actual compute cost.

**The real discovery: hand_landmark_10mb**

While searching PINTO0309's model zoo for alternatives, we found `hand_landmark_sparse_Nx3x224x224.onnx` (confusingly named "sparse" upstream despite being the LARGER model) -- a **bigger variant of the same MediaPipe hand landmark model** we already use. Same family, same architecture, same Apache 2.0 license. We call it **hand_landmark_10mb** (2,643,047 params) vs our current **hand_landmark_4mb** (1,011,718 params). Key differences:
- 10.1MB vs 3.9MB (wider layers, 2.6x more parameters)
- Supports batch inference: `[N, 3, 224, 224]` input (both hands in one forward pass)
- Same output format: `xyz_x21 [N, 63]`, `hand_score [N, 1]`, `lefthand_or_righthand [N, 1]`
- **Pinch tracking is VASTLY superior** to hand_landmark_4mb

Running through ORT WebGPU (not yet ported to our WGSL engine), hand_landmark_10mb does both hands batched in ~11ms. Through our fused WGSL engine, estimated ~5-8ms per hand.

**Test page:** `engine/test-rtmpose.html` has a dropdown to switch between all four models live. Uses `requestVideoFrameCallback` + fire-and-forget pattern matching ball-toss demo.

**Technical notes for the test page:**
- ORT loaded with pre-fetched wasm binary to bypass Vite's module transform (Vite rewrites ORT's internal dynamic imports, breaking wasm resolution)
- RTMPose uses ImageNet normalization (mean/std), NCHW format, 256x256 input, SimCC decode (argmax over [1,21,512] tensors)
- hand_landmark_10mb uses MediaPipe normalization ([0,1]), NCHW format, 224x224 input, same decode as hand_landmark_4mb
- Rotated-rect crop via canvas affine chain with projection in pixel space (not normalized space) to avoid aspect ratio distortion

### Next: port hand_landmark_10mb to WGSL engine

hand_landmark_10mb is the clear upgrade path. Same architecture as hand_landmark_4mb (which our WGSL engine already runs), just wider. The port:

1. Write `onnx_to_json.py` conversion script (doesn't exist yet -- `onnx` Python package is installed)
2. Dump the ONNX graph to JSON + extract weights to flat binary
3. Feed through `ModelRunner` -- should compile with zero new ops needed
4. Wire into `landmark-worker-wgsl.js` with a config flag for model selection
5. Benchmark through WGSL engine (expect ~5-8ms per hand, ~8-12ms batched)

**The batch dimension is the real win.** Currently our pipeline runs two separate landmark inference calls (one per hand). hand_landmark_10mb's `[N, 3, 224, 224]` input lets us do both in a single GPU dispatch chain. This alone could cut landmark latency nearly in half for two-hand tracking.

**Priority:** HIGH. This is the cheapest accuracy improvement available -- no new architecture, no new ops, same license. Just bigger weights through the same engine.

**Naming note:** The upstream ONNX file is `hand_landmark_sparse_Nx3x224x224.onnx` -- "sparse" is the MediaPipe internal codename, confusingly applied to the LARGER model. We use size-based names: **hand_landmark_4mb** (current, 1M params, 3.9MB) and **hand_landmark_10mb** (upgrade, 2.6M params, 10.1MB).

### Face tracking alternative: PFLD (Practical Facial Landmark Detector)

Lower priority than the RTMPose hand swap, but worth evaluating as a second face-tracking backend.

**Why consider it:** MediaPipe face (current, Apache 2.0, 478 3D points + 52 blendshapes) is genuinely hard to beat in the open-license tier -- face tracking has NOT been a pain point in any of our debugging sessions. PFLD (Practical Facial Landmark Detector) is Apache 2.0, lightweight, and fast, but only outputs ~98 2D points with no blendshapes. It's not an upgrade for Spellaria's face feature set, but it's a good candidate for a "fast minimal face tracker" backend for users who don't need the full face mesh.

**What PFLD gives us:**
- Apache 2.0 license, commercial OK
- ~98 2D landmarks (versus MediaPipe's 478)
- Very small model, very fast inference
- No blendshapes / expression coefficients
- 2D only, no z depth

**Use case:** demos or integrations where the consumer wants a lightweight face bounding-landmark tracker rather than the full mesh + blendshape pipeline. Could ship as a toggle in the same library: "Face: MediaPipe (full mesh + expressions)" or "Face: PFLD (fast, 98 points, 2D only)".

**Priority:** after RTMPose hand evaluation + z-depth + output adapters. This is a nice-to-have for demonstrating library flexibility, not a Spellaria blocker.

**Evaluation plan (when we get to it):**
1. Find a pre-exported PFLD ONNX model (several exist on GitHub / HuggingFace under Apache 2.0)
2. A/B test via ORT Web before plumbing through our WGSL engine (same approach as RTMPose)
3. If it runs and looks reasonable, wire through a `face-landmark-worker-wgsl.js` variant. Face detection stage (BlazeFace) stays the same -- PFLD expects a pre-cropped face.
4. Expose as a dropdown in the hub / ball-toss demo.

**Landscape of face-tracking models surveyed (all commercial-license-checked, 2026-04-17):**

| Model | License | Keypoints | Notes |
|---|---|---|---|
| MediaPipe Face Landmarker (current) | Apache 2.0 | 478 3D + 52 blendshapes | Best-in-class open-license, shipping |
| RTMW (whole-body) | Apache 2.0 | 133 total (face subset) | Face coupled to whole-body, big model |
| PFLD | Apache 2.0 | ~98 2D | Fast + small, no blendshapes |
| dlib shape_predictor_68 | Boost Software License | 68 2D | Old (2014), non-NN, poor accuracy |
| InsightFace (trap) | Code MIT, weights non-commercial | various | Commonly mistaken as commercial-OK; it is NOT |
| OpenFace | Academic-only | -- | Dead end for product |

### Z-depth via landmark spread (planned approach, not yet implemented)

When the time comes for real z-depth:

Rather than use the landmark model's noisy per-frame z regression, derive depth from the pixel distance between two anatomically-rigid landmarks. Wrist (idx 0) to middle-finger MCP (idx 9) is the ideal pair -- rigid palm bone structure, doesn't change when fingers flex.

```js
// Calibrate once: capture REFERENCE_PX_DIST at a known z (e.g. arm's length)
const dx = (landmarks[0].x - landmarks[9].x) * vw;
const dy = (landmarks[0].y - landmarks[9].y) * vh;
const pxDist = Math.hypot(dx, dy);
const depthNorm = REFERENCE_PX_DIST / pxDist;  // 1.0 = neutral, >1 = farther, <1 = closer
```

**Why this beats the model's z output:** landmark 2D coords are already noise-filtered (by our One Euro filter), so the derived depth is temporally smooth. Model z is regressed per-frame, noisy, and trained on synthetic depth data of varying quality.

**Caveats:**
- Out-of-plane rotation (palm tilting toward/away from camera face-on vs edge-on) changes the spread without real depth change. Acceptable for ball-toss where the hand is mostly approaching/retreating forward.
- Hand size varies by person. Auto-calibrate on first clean detection.

For ball-toss specifically, throw velocity can be derived from rate-of-change of the spread metric -- gives a physically meaningful "push toward camera" gesture.

### Future: output adapters + OSC pipeline

The library currently emits landmark/blendshape/gesture events in-process (consumed by the same page's JS). A natural extension: **pluggable output adapters** that let consumers route events to other processes, devices, or pages. Sketched but not built.

**Why it matters:** demos and installations are where hand tracking shines beyond a single game. "Point at the camera and trigger a sound in Ableton" is the pitch. Creative-coding community (TouchDesigner, Max/MSP, Wekinator users) expects this kind of I/O.

**Architecture:**

```js
const pipeline = new HandTracker();
const osc = new OSCOutput({ transport: 'websocket', url: 'ws://localhost:8080' });
pipeline.onGesture((g) => osc.send(`/spell/${g.name}`, g.confidence));
```

Library stays focused on inference. The adapter layer is optional; import only what you need.

**Browser constraint:** browsers can't send raw UDP (what OSC traditionally uses). Three viable routes:

1. **WebSocket → UDP relay**: tiny Node or Python script that accepts WebSockets and forwards as UDP OSC. Standard practice in browser-to-DAW integrations. Relay can run on the same machine or on a Pi on the local network. Phones connect via WebSocket. Latency: ~1-2ms locally.
2. **Direct WebSocket "pseudo-OSC"**: when both ends are browsers (another tab, phone's browser), skip UDP entirely. Serialize OSC binary over WebSocket. `osc-js` handles this natively.
3. **BroadcastChannel** for same-origin tabs. Zero config, 0ms latency, perfect for "trigger something on another page on this device".

**Recommended implementation order (when the time comes):**
1. Build a minimal `EventBus` in the library: `pipeline.on('handLandmarks', fn)`, `pipeline.on('gesture', fn)`, etc. Internal-only, no transport.
2. Define the `OutputAdapter` interface: `{ send(event, ...args) }`.
3. Implement adapters as separate, optional imports: `OSCOutput`, `BroadcastChannelOutput`, `MIDIOutput` (Web MIDI API), `WebhookOutput` (POST to URL), `PostMessageOutput` (iframes / other tabs).
4. Ship a reference UDP relay in `tools/osc-relay/` -- ~30 lines of Node using `dgram` + `ws`. Documented in README.
5. Demo: the ball-toss shows hand gestures triggering OSC to a hypothetical "DAW at localhost:8080". Optional, disabled by default.

**Library size cost**: `osc-js` is ~20KB. Only loaded when the OSC adapter is imported. Core library unaffected.

**Not a near-term priority**, but worth reserving the architectural space. When implementing, make sure the EventBus is designed such that adapters are truly plug-in (no core changes needed to add a new transport).

**Priority**: after RTMPose evaluation + z-depth. These are all research/exploration tasks that don't block shipping the core library.

### Very-future exploration: input adapters (Leap Motion, WebXR hand tracking)

Mirror of the output adapter layer: let the library accept hand/face data from sources OTHER than a webcam + our WGSL engine. Same event shape downstream -- gesture classifiers, z-depth derivation, OSC output all work regardless of where the landmarks came from.

**Candidates:**

- **Leap Motion / Ultraleap**: dedicated IR hand-tracking hardware (~$130). Orders of magnitude more accurate than any camera-based solution -- active infrared, purpose-built for hands, ~120fps, zero occlusion issues when hands overlap (the IR sees through). Browser integration via Ultraleap's legacy `leapjs` WebSocket daemon (`ws://localhost:6437`) which is still supported even in the newer Gemini/Hyperion SDKs. Desktop-only, requires installed drivers. Great as a "pro/installation mode" for demos, live performance, or Spellaria's high-end version.
- **WebXR hand tracking**: Meta Quest 3, Apple Vision Pro, and other headsets expose 25-joint hand skeletons via the [WebXR Hand Input API](https://immersive-web.github.io/webxr-hand-input/). Completely standard web API -- just listen for hand input on the XRSession. No driver install. Works inside the headset's browser. For any immersive VR/AR demo this is the right source.
- **External MediaPipe via bridge**: running MediaPipe's C++/Python pipeline on a beefier host and streaming landmarks to the browser via WebSocket. Useful for offloading compute or using models too large for browser inference.

**Architectural fit:**

```js
// camera-based (default, current)
const pipeline = new HandTracker();

// Leap Motion
const pipeline = new LeapHandTracker({ wsUrl: 'ws://localhost:6437' });

// WebXR (inside a VR/AR session)
const pipeline = new WebXRHandTracker(xrSession);
```

All three emit the same landmark event shape. Downstream consumers (gesture classification, z-depth estimation, OSC output, game code) never know or care which adapter produced the data. This is the same pluggable pattern as the output adapters.

**Why this matters long-term:** the library stops being "a WebGPU hand tracker" and becomes "a unified hand/pose pipeline for the browser, with multiple input and output backends." The WebGPU engine is the flagship backend (free, universal, runs anywhere), but the library's real value is the event pipeline and the classifier/adapter infrastructure built around it.

**Priority:** very-future exploration. No work until core library + RTMPose + z-depth + output adapters are done. Not on any near-term path.

### CRITICAL PATH: SpellARia Motion Signature Recorder

**This is the path to eliminating perceived latency in SpellARia.** Camera-plus-inference adds inherent lag to hand tracking -- by the time landmarks are rendered, the hand has already moved. A prediction layer can compensate by extrapolating forward, but naive prediction overshoots at motion edges (hand reverses direction, or stops). The current workaround -- backing off prediction whenever velocity drops -- trades responsiveness everywhere to avoid overshoot at edges.

The recorder exists to collect labeled data that lets us train a classifier to distinguish the kinematic signature of a REVERSAL (hand is about to change direction, keep predicting through zero-velocity) from a TERMINATION (hand is stopping, pull prediction back). With that classifier, SpellARia gets full-strength prediction during continuous motion AND correct behavior at motion edges -- the best of both worlds, and substantially lower perceived latency than MediaPipe or any competitor.

**This is why SpellARia can feel better than anything else on the market.** Not a research side project. It is the latency story.

#### Purpose
Collect labeled kinematic data to test whether reversals and terminations have distinguishable signatures in the frames leading up to zero-velocity. If they do, we can build a classifier that gates prediction behavior: maintain prediction through a detected reversal, pull prediction back on a detected termination. This would give SpellARia correct behavior at motion edges rather than forcing the current tradeoff of backing off prediction whenever velocity drops.

#### Protocol
- Hold R while performing a reversal. Release when done.
- Hold T while performing a termination. Release when done.
- Trial saved on keyup with label.

Keyup timing is not precision-critical. Alignment happens offline at analysis time against kinematic events in the data (zero-velocity for T, first-derivative sign change for R). The keyup just labels the trial.

#### Per-sample record
```javascript
{
  t: 127.3,
  keypoints: {
    wrist:     { x, y, z, confidence },
    palm:      { x, y, z, confidence },
    indexTip:  { x, y, z, confidence },
    middleTip: { x, y, z, confidence },
  },
  dropped: false,
}
```

Multiple keypoints because the biomechanical signature may show more cleanly in one than another. Wrist captures bulk arm motion with less noise. Fingertip captures fine motion but has its own articulation dynamics. Recording all of them is free at collection time and lets analysis explore which channel carries the signal best. `confidence` travels with each keypoint so analysis can reject low-quality samples without re-running the tracker.

`dropped: true` for frames where the tracker returned nothing. Preserves the true temporal structure -- a silently-omitted frame would distort derivative computations during analysis.

#### Per-trial record
```javascript
{
  trialId: "reversal_1729800000000_a3f",
  label: "reversal" | "termination",
  keydownTime: 1729800000000,
  keyupTime:   1729800000847,
  sessionId: "session_1729799000000_xyz",
  metadata: {
    handedness: "right",
    measuredFps: 29.8,
    trackerVersion: "webgpu-vision-0.3.1",
    smootherBypass: true,
  },
  samples: [ ],
}
```

`sessionId` lets analysis group trials recorded under the same conditions (same lighting, same camera setup, same warmup state). Cross-session variance can swamp class differences if you don't control for it.

`smootherBypass: true` is a runtime assertion that the recorder is actually reading pre-smoother data. If this ever shows false, the dataset is contaminated and should be discarded.

`measuredFps` because the nominal 30fps is often closer to 28-29fps with jitter, and derivatives need the actual frame rate to compute correctly.

#### Dataset format
JSON Lines. `spellaria_motion_dataset_YYYY-MM-DD_HHMMSS.jsonl`

JSONL streams trivially into Python/pandas for analysis, appends cleanly across sessions without rewriting a big JSON array, and survives partial writes if the browser crashes mid-export.

#### Data source
Subscribe to the hand tracking pipeline before the adaptive smoother. The whole experiment tests biomechanical signatures in raw motion. Recording smoothed data would measure the smoother's behavior, not the hand's.

#### UI

```
┌────────────────────────────────────────────┐
│ SpellARia Motion Signature Recorder        │
├────────────────────────────────────────────┤
│                                            │
│     [Live webcam + hand skeleton]          │
│                                            │
│  ● RECORDING REVERSAL  (0.34s)             │
│                                            │
│  Hold R → reversal                         │
│  Hold T → termination                      │
│  Backspace → discard last trial            │
│                                            │
│  Reversals:     14                         │
│  Terminations:  11                         │
│  FPS avg:       29.8                       │
│  Dropped frames: 3 of 2,847                │
│                                            │
│  [ Export Dataset ]  [ Clear Session ]     │
└────────────────────────────────────────────┘
```

Live skeleton overlay so you can confirm the tracker is locked onto your hand before starting a trial. Bad tracking produces bad data and should be caught at collection time, not at analysis time.

FPS and dropped-frame counts visible during collection so degrading tracking performance (lighting change, CPU spike) is noticed immediately rather than discovered in post-analysis.

Per-class counters so you can balance the dataset live rather than exporting, analyzing, realizing you have 40 reversals and 12 stops, and re-collecting.

Backspace discards the last trial -- for when you flub a recording (moved the wrong way, hand out of frame, realized mid-trial the class was wrong). Without this you either keep bad trials or pause to clean up IndexedDB manually.

Visual states:
- IDLE: white dot, "READY"
- RECORDING_REVERSAL: red pulsing dot, elapsed time
- RECORDING_TERMINATION: blue pulsing dot, elapsed time
- SAVED: green checkmark, counter increments

Color coding at a glance so you know which key you're actually holding without looking at the keyboard.

#### State machine
```
IDLE ──R down──► RECORDING_REVERSAL ──R up──► save ──► IDLE
IDLE ──T down──► RECORDING_TERMINATION ──T up──► save ──► IDLE
```

Edge cases:
- Other key pressed during active trial: ignored. Switching labels mid-trial would produce a trial whose second half has the wrong label.
- Window blur during recording: end trial, flag in metadata. A backgrounded tab throttles rAF and poisons the data.
- Tracker returns no frame: mark `dropped: true`, continue. Don't end the trial -- users shouldn't lose a good trial because of one frame hiccup.

#### File structure
```
/motion-recorder/
  index.html
  main.js              # entry, wires modules
  recorder.js          # trial state machine, key handling
  tracker-bridge.js    # subscribes to RAW hand tracking stream
  storage.js           # IndexedDB wrapper
  export.js            # .jsonl download
  ui.js                # DOM updates
  fps-monitor.js       # frame rate, drop detection
  styles.css
```

Modular so the tracker-bridge can be swapped when the hand tracking pipeline evolves without touching recording logic. FPS monitoring is separated because it needs to keep running even when no trial is active (baseline tracker health check).

#### Storage
IndexedDB, two object stores: `trials` (keyed on trialId), `sessions`.

IndexedDB over localStorage because trial datasets can get large (hundreds of trials × hundreds of frames × several keypoints) and localStorage has tight size limits.

Sessions stored separately so setup metadata (tracker version, camera config, lighting notes if you add them) lives once per session rather than being duplicated on every trial.

#### Export
Button reads all trials from IndexedDB, serializes as JSONL, triggers download.

Export-on-demand rather than auto-export because you may want to record across multiple sittings and export once at the end. IndexedDB persists across page reloads so in-progress sessions survive browser restarts.

#### Analysis alignment (reference)
- T trials: align to velocity magnitude minimum
- R trials: align to first velocity sign change on dominant axis

Computed offline from the recorded data. The recording protocol just produces labeled, time-stamped raw motion streams; extracting the alignment anchor is an analysis step.

#### Open questions (to resolve when building)
1. Which hook in the WebGPU hand tracking pipeline exposes raw pre-smoother keypoints? (Today the `HandTracker.processFrame()` result hands returns landmarks straight from the landmark worker -- no smoother inside the library. Smoothing happens in the demo's `handleHandResult` via the One Euro filter. So subscribe BEFORE `handleHandResult` runs filters -- or ideally, have the library expose an `onRawFrame` callback.)
2. Coordinate space of tracker output -- pixel, normalized, or 3D world? (Current pipeline: x/y are normalized [0,1], z is the raw landmark-model z-output. Record as-is; convert to whatever space analysis wants offline.)
3. Default keypoint set (wrist/palm/indexTip/middleTip), or different? (Start with those 4; palm = centroid of wrist + 3 MCPs. If palm isn't already a single landmark, compute it at record time. Fine-grained analysis can add more later from the raw 21-keypoint stream.)
4. Existing IndexedDB schema in SpellARia to avoid conflicts with, or greenfield? (Greenfield in this repo, under a namespaced DB name like `wgv_motion_recorder` to avoid collisions if this lib + Spellaria eventually run on the same origin.)

**Priority: critical path for SpellARia, near-term.** Build the recorder early. Data collection takes real wall-clock time (you have to physically perform hundreds of labeled motions), and every day it's delayed is a day of latency we could be eliminating. The recorder itself is a day of work; data collection is the longer tail; classifier training is straightforward once data exists.

Order of operations:
1. Build the recorder (consumes the existing `HandTracker.processFrame()` raw output; no library changes required).
2. Collect a balanced dataset (≥100 trials per class, multiple sessions, multiple lighting conditions).
3. Analyze offline (Python/pandas on the JSONL) to confirm the signatures are distinguishable before committing to classifier training.
4. If signatures exist: train a small classifier (MLP or LSTM on a sliding window of derivatives), integrate into the SpellARia prediction pipeline. If signatures don't exist: fall back to the current conservative approach and pivot to other latency wins (prediction horizon, smoother tuning, etc).

Worth doing in parallel with RTMPose evaluation -- the recorder consumes whatever pose source you hand it. If RTMPose lands later, the dataset can be re-collected or re-trained on the new source.

#### Spec addition: Onset Prediction

The motion-edge classifier (reversal vs termination) handles the END of motion. This addition handles the BEGINNING. Together they cover both sides of the perceived-latency problem.

##### Purpose
Detect motion initiation from rest and predict 1-2 frames forward on the emerging trajectory. Goal: visual hand starts moving on the same frame the user's real hand does, instead of lagging by the smoother's warmup window. Must reject tracker jitter to avoid amplifying non-motion into phantom movement.

##### New recording keys
- **Hold O** → onset-only trial. Hand starts at rest, moves. End of motion not tracked -- we only care about the initiation.
- **Hold J** → jitter baseline. Hand held still for the duration of the key press. Collects the null distribution of tracker noise at rest.

##### New trial labels
`label` field extends to: `"reversal" | "termination" | "onset" | "jitter"`

Everything else in the per-trial record is unchanged. J trials will tend to be short (1-2s), O trials short to medium (500ms-1s). No protocol change beyond label.

##### Target trial counts for pilot
- Reversals: 30
- Terminations: 30
- Onsets: 30
- Jitter: 20 (shorter trials, enough samples for a stable noise floor)

##### Detection features (computed at analysis time, then ported to runtime)

Two features gate onset detection:

**Feature 1: sustained acceleration magnitude**
```
mean_accel_mag = mean(|accel[n-2]|, |accel[n-1]|, |accel[n]|)
```
Averaged over 3 frames to reject single-frame jitter spikes. Computed per-axis then magnitude, or on the full 3D acceleration vector -- both worth testing.

**Feature 2: acceleration direction consistency**
```
direction_consistency = dot(accel[n], accel[n-1]) / (|accel[n]| * |accel[n-1]|)
```
Cosine similarity of consecutive acceleration vectors. Real motion produces consistent direction (near 1.0). Jitter produces random direction (near 0 or negative).

##### Threshold tuning
Thresholds are set empirically from the J dataset.

- `threshold_A` (magnitude): 95th percentile of `mean_accel_mag` across all jitter samples. Anything above this is above the noise floor.
- `threshold_B` (consistency): 95th percentile of `direction_consistency` across all jitter samples. Anything above this represents direction-coherent motion.

Validated against O dataset: check what fraction of true onsets are detected within 2 frames of the actual motion start (computed from velocity crossing a threshold). Aim for >90% detection rate with <5% false positives against the J dataset.

##### Runtime gate
```
if mean_accel_mag > threshold_A AND direction_consistency > threshold_B:
    fire_onset(predicted_position)
```

`predicted_position` = current smoothed position + current velocity * lookahead_ms. Lookahead starts at 1 frame (~33ms at 30fps). Bumping to 2 frames increases responsiveness but compounds false-positive cost -- tune after observing real behavior.

##### Pipeline integration

Onset prediction is a gated stage that runs only during the rest-to-motion transition. Once the adaptive smoother has accumulated enough trajectory history to produce stable output (existing behavior), onset stage goes dormant until the next rest interval.

```
Raw pose
   ↓
Onset detector
   ├─ Fires → output predicted position, persist for next 1-2 frames
   └─ Does not fire → pass raw through to adaptive smoother
   ↓
Adaptive smoother (unchanged)
   ↓
Display
```

State machine for the detector:
```
REST ──onset_detected──► PREDICTING ──hand_off──► TRACKING ──velocity<rest_threshold──► REST
```
- **REST**: velocity below rest threshold, running the onset gate every frame
- **PREDICTING**: onset detected, emitting predicted position for 1-2 frames while smoother spins up
- **TRACKING**: smoother has stable trajectory history, onset detector dormant

##### False-positive mitigations
- Require direction consistency across 2 consecutive frames before firing (not just one).
- Conservative lookahead (1 frame) until real-world false-positive rate is characterized.
- On false fire: predicted position snaps back to raw when velocity fails to sustain. Single-frame visual artifact, acceptable as worst case.

##### Open questions
1. Compute acceleration from wrist, palm, or fingertip keypoint? Likely wrist (least noisy), but worth testing across channels in analysis.
2. Should thresholds be global or per-user-session? Individual hand dynamics vary; per-session calibration (a 5-second "hold still" prompt at startup) may be worth the friction.
3. Interaction with reversal detector (when that lands): onset and reversal both involve acceleration from low-velocity state. Classifier may need to arbitrate.

### The original Phase 4 plan (still relevant for context)

### The realization

Phase 1.5 killed CPU readbacks **inside** the inference cascade. Camera frame -> letterbox -> detect -> warp -> landmark all stays on the GPU, all stages share an ONNX RT WebGPU device, every model-to-model handoff is via `Tensor.fromGpuBuffer()`. That is the headline win and it shipped.

But the workers' final output is **a JS `Float32Array` returned across the worker boundary**, which means the landmark coordinates get **read back from GPU to CPU at the very end** even when the consumer is going to turn around and draw them on screen — the most GPU-native operation imaginable. Worse, the existing demos then draw the wireframes via **Canvas2D** (`ctx.lineTo`), which is the slowest, most legacy 2D API the browser ships, in a project literally called WebGPU Vision. The whole identity of the library says "GPU never asks CPU for permission" and the very last step of the pipeline asks the CPU to draw lines.

In absolute perf terms on M1 Max this is microseconds, not milliseconds. It is not a bottleneck on current hardware. **But it is thematically incoherent**, it stops being free on weaker hardware (Chromebooks, phones), and it blocks the spec-pure version of the library from existing. Worth fixing as its own focused phase, after the user-facing work in Phases 2 and 3 is done.

### What "pure GPU" actually means

End to end, with no readbacks anywhere unless the consumer explicitly opts in:

```
camera frame (GPU texture)
  -> letterbox (compute, GPU buffer)
  -> detect inference (GPU tensor in/out)
  -> anchor decode + NMS (still CPU today, see option below)
  -> warp (compute, GPU buffer)
  -> landmark inference (GPU tensor in/out)
  -> landmark coords land in a GPU buffer
  -> bound directly as a vertex buffer to a render pipeline
  -> drawn via WebGPU render pass to the canvas
  -> swapchain present
```

Zero `mapAsync`, zero `Float32Array`, zero `ctx.lineTo`, zero `glReadPixels` anywhere.

### The worker contract change (the architectural piece)

Today every landmark worker postMessages a `Float32Array` of coordinates back to the main thread. That is the readback we need to make optional.

New contract: `init({ outputMode: 'cpu' | 'gpu' })`.

- **`'cpu'` (default, current behavior):** workers read landmarks back and post a `Float32Array`. Existing apps keep working with no changes. Apps that need landmarks in JS for gesture detection, parallax math, One Euro filtering, etc all stay on this path.
- **`'gpu'` (new):** workers do NOT read back. Instead they expose a `GPUBuffer` handle that the main thread (or another worker on the same shared device) can bind directly. The postMessage carries metadata (frame id, count, layout) but no coordinate data. The consumer uses the buffer in a render pipeline.

The hard part is that **`GPUBuffer` is not transferable across worker boundaries**. So `'gpu'` mode requires either:
- (a) all workers AND the renderer to share a single ONNX-RT-owned WebGPU device that lives in the main thread, with the landmark workers writing into a buffer the main thread also has a handle to, or
- (b) the workers expose a way for the main thread to import a buffer reference (some browsers allow this via `device.importExternalBuffer`-like APIs, but support is uneven).

Option (a) is the more portable path. It implies a refactor where the main thread owns the WebGPU device and the workers receive it via a structured-clone-able handle (or are kicked out entirely in favor of main-thread orchestration with off-thread inference via `OffscreenCanvas`). Either way, this is a real change to how `pipeline.js` and `face-pipeline.js` are wired, not a one-line patch.

### Three implementation options for the drawing side

Once landmarks live in a `GPUBuffer`, drawing becomes a pure WebGPU render pipeline. Three ways to wire it:

1. **Three.js `WebGPURenderer` for ball-toss specifically.** Three.js ships a WebGPU backend at `three/addons/renderers/webgpu/WebGPURenderer.js`. The ball-toss demo currently uses `WebGLRenderer` (the default). Switching to `WebGPURenderer` is a small change at the renderer instantiation, but the materials, lights, and post-processing the demo uses need to be checked for parity. Some features still lag the WebGL renderer. Once switched, the wireframe overlays become `THREE.LineSegments` / `THREE.Points` with a `BufferGeometry` whose vertex buffer **is the landmark `GPUBuffer` from the worker**. Zero copy. The whole visual stack is then WebGPU end-to-end. Lowest effort for the highest visible payoff in the showcase demo.

2. **Raw WebGPU render pipeline written by us.** ~100 lines of WGSL + JS. Vertex shader takes a uniform buffer of landmark positions, fragment shader does line + point rasterization. Used by the minimal demos (`/index.html`, `/face.html`) and exposed as a library helper `drawWireframeWebGPU(canvas, gpuBuffer, schema)`. Zero deps, zero Three.js, total control. The library-correct answer because it does not make webgpu-vision secretly require Three.js.

3. **Both — Three.js `WebGPURenderer` for ball-toss, raw WebGPU helper for the minimal demos and as a public library export.** Best of both. Ball-toss gets the full Three.js scene graph on WebGPU because it has a real 3D scene with lighting, shadows, projectiles. Library users who just want a drop-in wireframe overlay get the lightweight helper. This is the recommended path.

### Anchor decode + NMS: should they move to compute too?

Currently both the hand and face pipelines decode anchors and run weighted NMS on the **CPU**, after reading the detection model's output back as a `Float32Array`. That is a separate readback we have not addressed.

For "pure GPU" to be honest end-to-end, anchor decode and NMS need to be compute shaders too. Anchor decode is trivially parallel (one workgroup per anchor, multiply-add math). NMS is harder because it is order-dependent and weighted NMS specifically needs to average overlapping detections by score — there is published work on GPU NMS approaches (parallel reduction, sweep-and-prune, etc) but it is not free to implement.

Defer this until after the worker contract change lands. It is a meaningful piece of work on its own and it is downstream of the contract decision.

### Order of operations within Phase 4

1. Design and prototype the `outputMode: 'gpu'` worker contract on **one** worker (suggest: `landmark-worker.js`, since it is the simplest cascade endpoint). Confirm the cross-worker buffer sharing approach actually works on M1 Max Chrome, then on a couple of secondary platforms.
2. Land option 2 first (raw WebGPU wireframe helper), wired to that one worker, with one demo using it end-to-end. Smallest possible vertical slice. Validate that the GPU-only path is actually faster or at least at parity with the readback path (it should be, especially on weaker hardware).
3. Roll the contract change out to the other workers (palm, face detection, face landmark, face blendshape) once the pattern is proven.
4. Add option 1 (Three.js `WebGPURenderer` swap) for ball-toss.
5. Move anchor decode + NMS to compute shaders, removing the last detection-side readback.
6. Update the README and ARCHITECTURE.md to describe the pure-GPU path as the default and CPU readback as opt-in.

### Why this is Phase 4 and not "do it now"

- It is not a perf bottleneck on current hardware. Phase 2 (the unified comparison hub) and the Phase 3 polish items move the library from "works" to "people can actually use it." Phase 4 moves it from "works in spirit" to "works in spec." That ordering matches user impact.
- The worker contract change deserves careful design and a clean session, not a rushed implementation jammed between two other in-flight things.
- Doing Phase 4 before Phase 2 means Phase 2 (the hub) would have to be ported twice — once as Canvas2D, once as WebGPU. Doing Phase 2 first, then Phase 4, lets us do the WebGPU port once across all the demos at the same time.
- The per-mode settings work that the user originally asked for (and that we got happily distracted from when MediaPipe FaceLandmarker came up) needs to land first. That is the more pressing user-facing item.

When the next session takes this on: read this whole section, look at the **investigate why hand >> face** entry above (it informs which workers benefit most from going pure-GPU), and prototype on one worker before touching all of them.

## File inventory: what comes from where

### From `../3d-parallax-head-hand-tracking-demo/` into `webgpu-vision/`

| Source | Destination | Phase | Notes |
|---|---|---|---|
| `gpu-vision/src/palm-worker.js` | `src/palm-worker.js` | 1 | Has zero-copy GPU letterbox path |
| `gpu-vision/src/landmark-worker.js` | `src/landmark-worker.js` | 1 | |
| `gpu-vision/src/pipeline.js` | `src/pipeline.js` | 1 | Keeps `enableMemPattern: true` |
| `gpu-vision/src/face-detection-worker.js` | `src/face-detection-worker.js` | 1 | |
| `gpu-vision/src/face-landmark-worker.js` | `src/face-landmark-worker.js` | 1 | |
| `gpu-vision/src/face-blendshape-worker.js` | `src/face-blendshape-worker.js` | 1 | |
| `gpu-vision/models/palm_detection_lite.onnx` | `models/palm_detection_lite.onnx` | 1 | Do not copy `.backup` file |
| `index.html` | `demos/ball-toss/index.html` | 1 | Fix two `gpu-vision/src/` import paths |
| `coi-serviceworker.js` | `demos/ball-toss/coi-serviceworker.js` | 1 | |
| `face-worker.js` | `demos/ball-toss/face-worker.js` | 1 | MediaPipe backend worker |
| `hand-worker.js` | `demos/ball-toss/hand-worker.js` | 1 | MediaPipe backend worker |
| `mediapipe-vision.js` | `demos/ball-toss/mediapipe-vision.js` | 1 | MediaPipe loader shim |

### Files in `webgpu-vision/` that change

| File | Phase | Change |
|---|---|---|
| `src/*` (six files above) | 1 | Replaced with parallax repo versions |
| `models/palm_detection_lite.onnx` | 1 | Replaced |
| `README.md` | 1 | Add "Demos" section |
| `vite.config.js` | 1 if needed | Add `demos/ball-toss/index.html` to rollup input only if Vite does not auto-serve it |
| `index.html` | 2 | Replaced with one-stop hub |
| `face.html` | 2 | Deleted after hub verified |
| `README.md` | 2 | Update "Demos" section |

## Open questions parked for later

1. Model provenance (see Phase 3).
2. Final name for the ball-toss demo and folder.
3. Whether to land the benchmark UI inside the hub or keep it as a separate page.
4. Whether to publish the library to npm. If yes, the demo import paths will switch from `../../src/` to `webgpu-vision`.
