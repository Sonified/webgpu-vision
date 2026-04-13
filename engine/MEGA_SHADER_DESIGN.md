# Mega-Shader Fusion Design

Date: 2026-04-13
Status: Design only, not yet implemented

## The Problem

From `benchmark-baseline.md`: the face detector runs 53 dispatches at 12.70ms while ORT-CPU does it in 3.07ms. Dispatch overhead (~0.15ms each) accounts for ~8ms -- over 60% of total runtime. The GPU never gets a chance to stretch its legs because the CPU keeps interrupting it.

The analogy: imagine a jazz band where the conductor stops the music after every single measure to hand out the next page of sheet music. The musicians are great, but they spend more time waiting for paper than playing. What we want is to hand them the whole chart and let them play.

## The Insight

These models are built from a small vocabulary of repeating blocks:

**Face detector residual block** (3x3 DW convolutions):
```
DW Conv 3x3 -> 1x1 Conv -> [Pad channels?] -> Add (residual) -> ReLU
```

**Palm detector residual block** (5x5 DW convolutions):
```
DW Conv 5x5 -> 1x1 Conv -> [Pad channels?] -> Add (residual) -> PReLU
```

A "stage" is 2-4 of these blocks at the same spatial resolution, bookended by MaxPool (downsample) at the end or Resize (upsample) in the FPN head. The entire backbone is just stages stacked.

The key move: instead of the CPU telling the GPU "do this conv, now do this conv, now do this add," we pack a descriptor table into a uniform buffer and let a single shader loop through the whole block internally, reading weights from a consolidated buffer using offsets from the descriptor.

## Fusion Levels

### Level 1: Residual Block Fusion (the primary target)

Fuse: DW Conv + 1x1 Conv + Add + Activation = 1 dispatch instead of 3-4.

This is the money move. Each model has 16-20 residual blocks. Going from ~3 dispatches per block to 1 dispatch cuts the backbone from ~48-60 dispatches to ~16-20.

### Level 2: Stage Fusion (stretch)

Fuse: multiple residual blocks at the same spatial resolution into a single dispatch.

This requires storing intermediate activations in workgroup-shared memory or private registers between blocks. Feasibility depends on spatial dimensions and channel counts -- see Risks section.

### Level 3: Backbone Fusion (aspirational, probably not practical)

Fuse an entire backbone stage + MaxPool. The MaxPool changes spatial dimensions, which means the workgroup grid geometry changes mid-computation. Not impossible (the shader can internally re-map thread IDs) but complex and likely not worth the effort given Level 1 already gets us most of the win.

## Shader Structure (Pseudocode)

### Level 1: Fused Residual Block

```
// One dispatch per residual block.
// Thread mapping: one thread per output spatial position (oh, ow).
// The z-dimension of the workgroup handles output channels of the FINAL output.
// Internally, the shader loops over intermediate channels.

struct BlockDescriptor {
    // DW conv params
    dw_in_channels: u32,
    dw_kern_size: u32,       // 3 or 5
    dw_stride: u32,          // 1 or 2
    dw_padding: u32,
    dw_weight_offset: u32,   // offset into consolidated weight buffer
    dw_bias_offset: u32,

    // 1x1 conv params
    pw_out_channels: u32,
    pw_weight_offset: u32,
    pw_bias_offset: u32,

    // Spatial dims
    in_h: u32,
    in_w: u32,
    out_h: u32,
    out_w: u32,

    // Residual connection
    has_residual: u32,       // 0 = none, 1 = same-channel add, 2 = needs channel padding
    residual_channels: u32,  // input channels to the residual (before padding)

    // Activation
    activation_type: u32,    // 0=none, 1=PReLU, 2=ReLU6, 3=ReLU
    activation_offset: u32,  // offset to PReLU slopes in weight buffer

    // For channel-padding residual path
    pad_channels_out: u32,   // padded channel count (0 if no padding needed)
}

@group(0) @binding(0) var<uniform> desc: BlockDescriptor;
@group(0) @binding(1) var<storage, read> input: array<f32>;       // block input
@group(0) @binding(2) var<storage, read> weights: array<f32>;     // ALL weights, indexed by offset
@group(0) @binding(3) var<storage, read> residual_in: array<f32>; // residual path input (may == input)
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let ow = gid.x;
    let oh = gid.y;
    let oc = gid.z;  // output channel of the 1x1 conv

    if (ow >= desc.out_w || oh >= desc.out_h || oc >= desc.pw_out_channels) { return; }

    // --- Phase 1: Depthwise convolution ---
    // Each output channel of the 1x1 conv reads from ALL dw output channels.
    // We can't store the full DW output in shared memory (too big for high channel counts),
    // so we interleave: for each DW output channel, compute DW result, immediately
    // accumulate into the 1x1 conv sum.

    var pw_sum: f32 = weights[desc.pw_bias_offset + oc];

    for (var ic: u32 = 0u; ic < desc.dw_in_channels; ic++) {
        // Compute DW conv for channel ic at position (oh, ow)
        var dw_sum: f32 = weights[desc.dw_bias_offset + ic];

        let kern = desc.dw_kern_size;
        for (var kh: u32 = 0u; kh < kern; kh++) {
            for (var kw: u32 = 0u; kw < kern; kw++) {
                let ih = oh * desc.dw_stride + kh - desc.dw_padding;
                let iw = ow * desc.dw_stride + kw - desc.dw_padding;
                if (ih < desc.in_h && iw < desc.in_w) {
                    let in_idx = ic * desc.in_h * desc.in_w + ih * desc.in_w + iw;
                    let w_idx = desc.dw_weight_offset + ic * kern * kern + kh * kern + kw;
                    dw_sum += input[in_idx] * weights[w_idx];
                }
            }
        }
        // No activation between DW and 1x1 in these models -- DW output feeds directly to 1x1

        // --- Phase 2: Accumulate into 1x1 conv ---
        let pw_w_idx = desc.pw_weight_offset + oc * desc.dw_in_channels + ic;
        pw_sum += dw_sum * weights[pw_w_idx];
    }

    // --- Phase 3: Residual add ---
    if (desc.has_residual >= 1u) {
        let spatial_idx = oh * desc.out_w + ow;
        if (desc.has_residual == 2u) {
            // Channel-padded residual: only add if oc < residual_channels
            if (oc < desc.residual_channels) {
                pw_sum += residual_in[oc * desc.out_h * desc.out_w + spatial_idx];
            }
            // else: oc >= residual_channels, residual contributes 0 (channel padding)
        } else {
            pw_sum += residual_in[oc * desc.out_h * desc.out_w + spatial_idx];
        }
    }

    // --- Phase 4: Activation ---
    if (desc.activation_type == 1u) {
        if (pw_sum < 0.0) { pw_sum *= weights[desc.activation_offset + oc]; }
    } else if (desc.activation_type == 2u) {
        pw_sum = clamp(pw_sum, 0.0, 6.0);
    } else if (desc.activation_type == 3u) {
        pw_sum = max(pw_sum, 0.0);
    }

    // --- Write output ---
    output[oc * desc.out_h * desc.out_w + oh * desc.out_w + ow] = pw_sum;
}
```

### Key Design Decision: Fused DW+PW Inner Loop

The critical trick is **not materializing the DW output to memory**. Instead of:
1. DW Conv -> write intermediate buffer -> barrier -> 1x1 Conv reads intermediate

We do:
1. For each DW channel, compute DW result in registers, immediately multiply by the 1x1 weight and accumulate

This eliminates the intermediate buffer entirely. The DW conv output for one spatial position across all channels lives in a single `f32` register (`dw_sum`) that gets reused each iteration. The 1x1 accumulator (`pw_sum`) is another single register.

Total register pressure per thread: 2 floats + loop counters. Negligible.

## Block Descriptor Uniform Layout

```
// 18 u32s = 72 bytes per block descriptor
struct BlockDescriptor {
    dw_in_channels:    u32,   // offset 0
    dw_kern_size:      u32,   // offset 4    (3 or 5)
    dw_stride:         u32,   // offset 8    (1 or 2)
    dw_padding:        u32,   // offset 12
    dw_weight_offset:  u32,   // offset 16   (float offset into weight buffer)
    dw_bias_offset:    u32,   // offset 20
    pw_out_channels:   u32,   // offset 24
    pw_weight_offset:  u32,   // offset 28
    pw_bias_offset:    u32,   // offset 32
    in_h:              u32,   // offset 36
    in_w:              u32,   // offset 40
    out_h:             u32,   // offset 44
    out_w:             u32,   // offset 48
    has_residual:      u32,   // offset 52   (0=none, 1=same-ch, 2=pad-ch)
    residual_channels: u32,   // offset 56
    activation_type:   u32,   // offset 60   (0=none, 1=PReLU, 2=ReLU6, 3=ReLU)
    activation_offset: u32,   // offset 64   (float offset for PReLU slopes)
    pad_channels_out:  u32,   // offset 68
}
```

All weight/bias/slope data lives in a single large `array<f32>` storage buffer. The descriptor tells the shader where to find each piece. No per-dispatch buffer creation.

## Dispatch Plan Per Model

### Face Detector (128x128 input, currently 53 dispatches)

```
Dispatch 1:  Initial Conv 5x5 stride 2 + ReLU          (standalone, not a residual block)
Dispatch 2:  Residual block 1 (DW 3x3 + 1x1 + Add + ReLU)   stage 1, 64x64
Dispatch 3:  Residual block 2 (DW 3x3 + 1x1 + Pad + Add + ReLU)  stage 1, 64x64
Dispatch 4:  MaxPool 2x2 + Pad channels               (stage boundary)
Dispatch 5:  Residual block 3 (DW 3x3 s2 + 1x1 + Pad + Add + ReLU)  stage 2 entry
Dispatch 6:  Residual block 4                           stage 2, 32x32
Dispatch 7:  Residual block 5 + Pad + Add + ReLU        stage 2, 32x32
Dispatch 8:  MaxPool + Pad                              (stage boundary)
Dispatch 9:  Residual block 6 (DW s2 + 1x1 + Pad)      stage 3 entry
Dispatch 10: Residual block 7                           stage 3, 16x16
... (continue pattern)
Dispatch ~14: Last backbone residual block
Dispatch 15: MaxPool (into 8x8 branch)
Dispatch 16-17: 8x8 residual blocks
Dispatch 18: Output conv heads (regressors + classificators, 2 dispatches)

Target: ~18 dispatches (down from 53)
```

Wait -- with Level 1 fusion only, the breakdown is:
- 1 initial conv (standalone)
- ~20 residual blocks -> 20 dispatches (each fusing DW+1x1+Add+Act)
- 3 MaxPool+Pad ops
- 4 output head convs + Transpose/Reshape/Concat

That's roughly **28 dispatches** -- a 47% reduction. Still meaningful but not the 5-8 target.

To hit 5-8, we need Level 2 (stage fusion) at least partially.

### Revised target with selective Level 2

If we fuse 2-4 residual blocks per stage into one dispatch (using register-only intermediate storage):

```
Dispatch 1:  Initial Conv 5x5 + ReLU
Dispatch 2:  Stage 1 (2 residual blocks at 64x64, 24-28 channels)
Dispatch 3:  MaxPool + Pad (stage boundary)
Dispatch 4:  Stage 2 (3 residual blocks at 32x32, 28-48 channels)
Dispatch 5:  MaxPool + Pad
Dispatch 6:  Stage 3 (6 residual blocks at 16x16, 48-96 channels)
Dispatch 7:  Stage 4 / 8x8 blocks (5 blocks at 8x8, 96 channels)
Dispatch 8:  Output heads (conv + transpose + reshape + concat)

Target: 8 dispatches (down from 53)
```

### Palm Detector (192x192 input, currently 64 dispatches)

```
Dispatch 1:  Initial Conv 5x5 s2 + PReLU               96x96, 32ch
Dispatch 2:  Stage 1 (3 blocks at 96x96, 32ch, 5x5 DW)
Dispatch 3:  MaxPool + Pad (96x96 -> 48x48)
Dispatch 4:  Stage 2 (4 blocks at 48x48, 64ch)
Dispatch 5:  MaxPool + Pad (48x48 -> 24x24)
Dispatch 6:  Stage 3 (4 blocks at 24x24, 128ch)
Dispatch 7:  MaxPool + Pad (24x24 -> 12x12)
Dispatch 8:  Stage 4 (4 blocks at 12x12, 256ch)
Dispatch 9:  FPN upsample + merge (Resize + Conv + Add)
Dispatch 10: FPN blocks at 24x24
Dispatch 11: FPN upsample + merge
Dispatch 12: FPN blocks at 48x48 (if present -- model may vary)
Dispatch 13: Output heads

Target: ~10-13 dispatches (down from 64)
```

## Level 2 Stage Fusion: How It Works

For a stage with N residual blocks at spatial size HxW with C channels:

```
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let ow = gid.x;
    let oh = gid.y;
    let oc = gid.z;

    // Read input value for this thread's position
    // (each thread is responsible for one (oh, ow, oc) output)

    for (var block = 0u; block < num_blocks; block++) {
        // Load block descriptor from uniform array
        let desc = block_descriptors[block];

        // Fused DW+1x1 for this block (same code as Level 1)
        var pw_sum = ...;  // compute as above

        // Residual + activation
        ...

        // The output of this block becomes the input of the next.
        // BUT: we can't read our own output -- other threads at different
        // spatial positions wrote values we might need for the NEXT block's
        // DW conv (which has a spatial kernel that reads neighbors).
        //
        // This is the fundamental problem with stage fusion.
        // Options:
        //   A) storageBarrier() between blocks (NOT available in WGSL for cross-workgroup sync)
        //   B) Workgroup-shared memory scratchpad (only works within one workgroup)
        //   C) Write to global memory + separate dispatch (defeats the purpose)
    }
}
```

### The Cross-Workgroup Barrier Problem

WGSL has `storageBarrier()` for workgroup-internal sync but **no global barrier between workgroups**. A DW conv with kernel size K reads from spatial neighbors up to K/2 pixels away. If those neighbors belong to a different workgroup, we cannot guarantee they have finished writing before we read.

### Solution: Tile-Based Stage Fusion with Halo

Each workgroup processes a tile of spatial positions with a **halo** (ghost zone) that extends beyond the tile boundary by the kernel radius. The workgroup:

1. Loads the tile + halo from global memory into workgroup shared memory
2. Computes DW conv using shared memory (no cross-workgroup reads)
3. Computes 1x1 conv + residual + activation
4. Writes the non-halo portion to shared memory for the next block
5. Repeats for each block in the stage

The halo for a 5x5 kernel is 2 pixels on each side. For an 8x8 tile, we need 12x12 shared memory per channel. For 3x3 kernels, the halo is 1 pixel: 10x10 per channel.

**Shared memory budget** (WebGPU spec guarantees 16KB per workgroup):
- Stage at 64x64 with 24 channels, 10x10 tile+halo: 10 * 10 * 24 * 4 = 9,600 bytes. Fits.
- Stage at 32x32 with 48 channels, 10x10 tile+halo: 10 * 10 * 48 * 4 = 19,200 bytes. Does NOT fit.
- Stage at 32x32 with 48 channels, 12x12 tile+halo (5x5 kernel): 12 * 12 * 48 * 4 = 27,648 bytes. Way over.

So tile-based stage fusion only works for early stages with low channel counts. Later stages (48+ channels) exceed the 16KB workgroup memory limit.

### Practical Level 2 Strategy

- **Early stages (up to ~32 channels)**: full stage fusion with shared memory scratchpad
- **Later stages (48+ channels)**: Level 1 only (one dispatch per residual block)
- **FPN head**: each Resize+Conv+Add is one fused dispatch

This gives a realistic target of:
- Face detector: ~12-15 dispatches (initial + 2-3 fused stages + individual blocks + output)
- Palm detector: ~15-18 dispatches (more blocks, higher channel counts limit fusion)

Still a major win (3-4x dispatch reduction) even without achieving the 5-8 target everywhere.

## Alternative to Level 2: Pre-Allocated Pipeline

Instead of chasing stage fusion, we can get most of the overhead reduction by fixing the CPU-side dispatch machinery:

1. **Pre-allocate all uniform buffers** at init time (currently creating new GPUBuffers per frame)
2. **Pre-create all bind groups** at init time
3. **Single command encoder** with all compute passes pre-recorded (currently already done, but with per-dispatch buffer creation overhead)

This won't reduce dispatch count but will cut per-dispatch overhead from ~0.15ms to ~0.02ms (eliminating buffer creation and bind group creation from the hot path). Combined with Level 1 fusion (cutting dispatches roughly in half), this could match or beat the 5-8 dispatch target in effective performance.

## Risks and Limitations

### Register Pressure in Level 1
The fused DW+1x1 inner loop has one accumulator (`pw_sum`) and one temporary (`dw_sum`). Minimal register pressure. The outer loop over `dw_in_channels` is sequential, not parallel, so no register blowup. This is safe.

### Channel Count Scaling
For the palm detector's later stages (256 channels, 5x5 DW kernel), the inner loop is:
```
for ic in 0..256:
    for kh in 0..5:
        for kw in 0..5:
            // 2 memory reads + 1 multiply-accumulate
```
That's 256 * 25 = 6,400 iterations per thread. Lots of work, but it's compute-bound (good for GPU) rather than overhead-bound (bad). The current approach does the same work but spreads it across 3 dispatches with intervening overhead. Fusing it does not increase total work, just consolidates it.

### Workgroup Shared Memory (16KB limit)
As analyzed above, the 16KB limit constrains tile-based stage fusion to low-channel-count stages. This is the main reason full backbone fusion (Level 3) is impractical. Level 1 fusion does not use shared memory at all, so it is unaffected.

### Divergent Thread Execution
The `if (ih < desc.in_h && iw < desc.in_w)` bounds check inside the kernel loop creates divergent execution at tile boundaries. This is identical to the current conv2d.wgsl behavior and is unavoidable for any padded convolution. No regression.

### Weight Buffer Size
All weights need to live in a single storage buffer so the shader can index by offset. Current sizes:
- Palm detector: 3.84MB (960K floats)
- Face detector: 0.40MB (101K floats)

Both well within WebGPU's `maxStorageBufferBindingSize` (128MB minimum guaranteed). Already implemented this way (the graph JSON has offsets into a consolidated weight buffer).

### MaxPool Cannot Be Fused Into Residual Blocks
MaxPool changes spatial dimensions (2x downsample). The workgroup grid is dispatched based on the OUTPUT spatial size. A fused block that includes MaxPool would need to dispatch for the larger (pre-pool) size for the blocks before the pool, and the smaller (post-pool) size for blocks after. This would waste threads (half the workgroup would be idle after the pool). Better to keep MaxPool as separate dispatches -- there are only 3-4 per model.

### Output Heads Require Transpose
The output conv heads feed into Transpose + Reshape + Concat, which currently does CPU readback (the ugliest part of model-runner.js). The mega-shader design does not address this -- it is orthogonal. A future WGSL transpose shader would eliminate the readback, but that is a separate task.

## Comparison to Current Approach

| Aspect | Current (naive) | Level 1 Fusion | Level 1 + Pre-alloc |
|---|---|---|---|
| Dispatches (face det) | 53 | ~28 | ~28 |
| Dispatches (palm det) | 64 | ~35 | ~35 |
| Per-dispatch overhead | ~0.15ms (buffer creation) | ~0.15ms | ~0.02ms (pre-allocated) |
| Total overhead (face) | ~8ms | ~4.2ms | ~0.6ms |
| Total overhead (palm) | ~9.6ms | ~5.3ms | ~0.7ms |
| Compute time (face) | ~4.7ms | ~4.7ms (same work) | ~4.7ms |
| Compute time (palm) | ~9ms | ~9ms | ~9ms |
| Expected total (face) | 12.7ms | ~8.9ms | ~5.3ms |
| Expected total (palm) | 18.6ms | ~14.3ms | ~9.7ms |
| Intermediate buffers | Many small | Eliminated within blocks | Eliminated within blocks |
| Code complexity | Simple | Moderate | Moderate |

**Key takeaway**: Level 1 fusion alone is worth ~30% speedup. Combined with pre-allocated pipelines, we should hit ~55-60% speedup -- enough to beat ORT-GPU on all three models, not just palm detection.

## Implementation Order

1. **Pre-allocate uniform buffers and bind groups** (low risk, immediate payoff, no shader changes)
2. **Level 1 residual block fusion shader** (the design above, moderate effort)
3. **Graph analyzer** that identifies residual blocks in the JSON graph and emits BlockDescriptors
4. **Fused MaxPool+Pad shader** (already partially done in maxpool.wgsl)
5. **Benchmark** against baseline
6. **Level 2 stage fusion** for early stages only (if Level 1 + pre-alloc doesn't hit targets)

Step 1 is the quickest win and should be done first regardless. It is also independently useful even if the mega-shader never ships.
