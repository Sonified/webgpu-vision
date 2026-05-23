# Start Failure Log

Tracking instances where the ball-toss demo fails to fully start.

---

## Issue: Tracking never engages on cold start (tab-away-and-back fixes it)

### Symptom

The demo loads. The 3D scene renders its initial frame -- cubes, background, everything looks normal. But the scene is **completely frozen**. The cubes are not rotating (they should be), the world clock is not advancing, no camera self-view appears, no tracking, nothing interactive. It's a static snapshot, not a living scene. The `animate()` rAF loop is either not running or dying on its first frame.

### The fix (every time)

Tab away from the page to another browser tab, then tab back. Tracking immediately starts working. This is 100% reproducible as a workaround -- it fixes the issue every single time.

### What this tells us

The `visibilitychange` handler fires when the tab is re-shown. It does three things:
1. Flushes stale `clock.getDelta()` 
2. Restarts `requestAnimationFrame(animate)`
3. Re-registers `requestVideoFrameCallback` for the active backend

Whatever is broken on cold start, the visibility change handler's restart is sufficient to fix it. This points to a race condition in the initial startup sequence -- something about the order or timing of camera-ready vs. animate loop vs. VFC chain vs. model init doesn't land correctly on first load, but the clean re-kick from visibilitychange sidesteps it.

### Console signature (when broken)

Everything *looks* healthy in the logs. All workers init and compile:

```
[palm-worker-wgsl] ready (compiled WGSL engine)
[landmark-worker-wgsl] ready (compiled WGSL engine)  (x2)
[face-detection-worker-wgsl] ready (compiled WGSL engine)
[face-lm-worker-wgsl] ready (compiled WGSL engine)
[blendshape-worker] ready
[lifecycle] All workers ready -- main thread is pure orchestration
[lifecycle] All face workers ready (1 face slots)
```

Camera goes live:

```
[lifecycle] startCamera()
[lifecycle] camera live (480x360)
```

The `wgpuVfcTick` fires and the animate loop runs. But the palm detector either:
- Returns 0 detections forever (letterbox dump shows ALL ZEROS in the input buffer), or
- Finds hands but the dedup logic rejects every detection

Either way, nothing ever makes it to the screen.

### Second log (2026-04-24): partial detection but still broken

A second occurrence showed the letterbox input was all zeros on the first frame (`[LB-DUMP] R channel: min=0.000 max=0.000 mean=0.000`), confirming the `VideoFrame` is being created before the camera has delivered real pixel data. After ~6 attempts, real detections started appearing, but everything was still visually dead -- the scene never came alive until tab-away-and-back.

### Not yet investigated

- Why the `visibilitychange` restart works but the initial startup doesn't -- they should be doing the same thing
- Whether the issue is the rVFC chain not actually starting, the animate loop not rendering camera/tracking output, or the self-view draw path never being triggered
- Whether this correlates with specific browsers, GPU load, or other WebGPU tabs being open
- Frequency: happens intermittently but often enough to be annoying

### Relation to other known issues

This is distinct from the "intermittent startup stall" in WORK-PLAN.md (where everything freezes for 20-30s then recovers with a burst). In this issue, the system never recovers on its own -- only the tab-away workaround fixes it.
