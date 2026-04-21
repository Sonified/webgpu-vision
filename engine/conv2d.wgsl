// Fused Conv2D compute shader.
// Handles depthwise, 1x1 pointwise, and general convolution.
//
// Optimizations:
// - Workgroup shared memory tile for depthwise conv
// - Manually unrolled 3x3 and 5x5 DW kernels
// - 1x1 path: each thread computes oc_tile output channels (4x fewer workgroups)
// - Double-unrolled vec4 accumulation (8 input channels per iteration)
// - Residual add fused into output write

struct ConvParams {
    batch: u32,
    in_c: u32,
    in_h: u32,
    in_w: u32,
    out_c: u32,
    out_h: u32,
    out_w: u32,
    kern_h: u32,
    kern_w: u32,
    stride_h: u32,
    stride_w: u32,
    pad_top: u32,
    pad_left: u32,
    group: u32,
    has_prelu: u32,
    has_residual: u32,
    oc_tile: u32,
}

@group(0) @binding(0) var<uniform> params: ConvParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> prelu_slope: array<f32>;
@group(0) @binding(5) var<storage, read> residual: array<f32>;
@group(0) @binding(6) var<storage, read_write> output: array<f32>;

var<workgroup> tile: array<f32, 400>;

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wgid: vec3u
) {
    let ow = gid.x;
    let oh = gid.y;
    let oc = gid.z;

    let channels_per_group = params.in_c / params.group;
    let group_id = oc / (params.out_c / params.group);
    let in_c_start = group_id * channels_per_group;
    let spatial_ok = ow < params.out_w && oh < params.out_h;

    var sum: f32 = 0.0;
    let in_bounds = spatial_ok && oc < params.out_c;
    if (in_bounds) { sum = bias[oc]; }

    // === DEPTHWISE PATH: shared memory tile + unrolled kernels ===
    if (params.group == params.in_c && params.kern_h <= 5u && params.kern_w <= 5u) {
        let tile_w: u32 = 8u * params.stride_w + params.kern_w - 1u;
        let tile_h: u32 = 8u * params.stride_h + params.kern_h - 1u;
        let origin_h: i32 = i32(wgid.y * 8u * params.stride_h) - i32(params.pad_top);
        let origin_w: i32 = i32(wgid.x * 8u * params.stride_w) - i32(params.pad_left);
        let local_idx = lid.y * 8u + lid.x;
        let tile_size = tile_w * tile_h;
        let in_c_idx = oc;

        for (var t = local_idx; t < tile_size; t += 64u) {
            let ty = i32(t / tile_w);
            let tx = i32(t % tile_w);
            let iy = origin_h + ty;
            let ix = origin_w + tx;
            var val: f32 = 0.0;
            if (iy >= 0 && iy < i32(params.in_h) && ix >= 0 && ix < i32(params.in_w)) {
                val = input[in_c_idx * params.in_h * params.in_w + u32(iy) * params.in_w + u32(ix)];
            }
            tile[t] = val;
        }
        workgroupBarrier();

        if (in_bounds) {
            let lo = lid.y * params.stride_h;
            let lw = lid.x * params.stride_w;
            let tw = tile_w;
            let wb = oc * params.kern_h * params.kern_w;

            if (params.kern_h == 3u && params.kern_w == 3u) {
                let b = lo * tw + lw;
                sum += tile[b]            * weights[wb];
                sum += tile[b + 1u]       * weights[wb + 1u];
                sum += tile[b + 2u]       * weights[wb + 2u];
                sum += tile[b + tw]       * weights[wb + 3u];
                sum += tile[b + tw + 1u]  * weights[wb + 4u];
                sum += tile[b + tw + 2u]  * weights[wb + 5u];
                let b2 = b + tw * 2u;
                sum += tile[b2]           * weights[wb + 6u];
                sum += tile[b2 + 1u]      * weights[wb + 7u];
                sum += tile[b2 + 2u]      * weights[wb + 8u];
            } else if (params.kern_h == 5u && params.kern_w == 5u) {
                let b = lo * tw + lw;
                sum += tile[b]            * weights[wb];
                sum += tile[b + 1u]       * weights[wb + 1u];
                sum += tile[b + 2u]       * weights[wb + 2u];
                sum += tile[b + 3u]       * weights[wb + 3u];
                sum += tile[b + 4u]       * weights[wb + 4u];
                let r1 = b + tw;
                sum += tile[r1]           * weights[wb + 5u];
                sum += tile[r1 + 1u]      * weights[wb + 6u];
                sum += tile[r1 + 2u]      * weights[wb + 7u];
                sum += tile[r1 + 3u]      * weights[wb + 8u];
                sum += tile[r1 + 4u]      * weights[wb + 9u];
                let r2 = b + tw * 2u;
                sum += tile[r2]           * weights[wb + 10u];
                sum += tile[r2 + 1u]      * weights[wb + 11u];
                sum += tile[r2 + 2u]      * weights[wb + 12u];
                sum += tile[r2 + 3u]      * weights[wb + 13u];
                sum += tile[r2 + 4u]      * weights[wb + 14u];
                let r3 = b + tw * 3u;
                sum += tile[r3]           * weights[wb + 15u];
                sum += tile[r3 + 1u]      * weights[wb + 16u];
                sum += tile[r3 + 2u]      * weights[wb + 17u];
                sum += tile[r3 + 3u]      * weights[wb + 18u];
                sum += tile[r3 + 4u]      * weights[wb + 19u];
                let r4 = b + tw * 4u;
                sum += tile[r4]           * weights[wb + 20u];
                sum += tile[r4 + 1u]      * weights[wb + 21u];
                sum += tile[r4 + 2u]      * weights[wb + 22u];
                sum += tile[r4 + 3u]      * weights[wb + 23u];
                sum += tile[r4 + 4u]      * weights[wb + 24u];
            } else {
                for (var kh: u32 = 0u; kh < params.kern_h; kh++) {
                    for (var kw: u32 = 0u; kw < params.kern_w; kw++) {
                        let tile_idx = (lo + kh) * tw + (lw + kw);
                        let w_idx = wb + kh * params.kern_w + kw;
                        sum += tile[tile_idx] * weights[w_idx];
                    }
                }
            }
        }
        // DW epilogue (shared with general path below)
    }
    else if (!spatial_ok) {
        return;
    }
    // === 1x1 POINTWISE PATH: multi-OC tiled, double-unrolled vec4 ===
    else if (params.kern_h == 1u && params.kern_w == 1u) {
        let spatial_idx = oh * params.in_w + ow;
        let cpg = channels_per_group;
        let stride = params.in_h * params.in_w;
        let out_stride = params.out_h * params.out_w;
        let oc_base = gid.z * params.oc_tile;

        for (var oc_t: u32 = 0u; oc_t < params.oc_tile; oc_t++) {
            let cur_oc = oc_base + oc_t;
            if (cur_oc >= params.out_c) { break; }

            var s: f32 = bias[cur_oc];
            let w_base = cur_oc * cpg;

            let cpg8 = cpg / 8u;
            for (var ic8: u32 = 0u; ic8 < cpg8; ic8++) {
                let ic = in_c_start + ic8 * 8u;
                let i0 = ic * stride + spatial_idx;
                let v0 = vec4f(input[i0], input[i0 + stride], input[i0 + stride * 2u], input[i0 + stride * 3u]);
                let w0 = vec4f(weights[w_base + ic8 * 8u], weights[w_base + ic8 * 8u + 1u], weights[w_base + ic8 * 8u + 2u], weights[w_base + ic8 * 8u + 3u]);
                let i1 = i0 + stride * 4u;
                let v1 = vec4f(input[i1], input[i1 + stride], input[i1 + stride * 2u], input[i1 + stride * 3u]);
                let w1 = vec4f(weights[w_base + ic8 * 8u + 4u], weights[w_base + ic8 * 8u + 5u], weights[w_base + ic8 * 8u + 6u], weights[w_base + ic8 * 8u + 7u]);
                s += dot(v0, w0) + dot(v1, w1);
            }
            let done8 = cpg8 * 8u;
            let cpg4_rem = (cpg - done8) / 4u;
            for (var ic4: u32 = 0u; ic4 < cpg4_rem; ic4++) {
                let ic = in_c_start + done8 + ic4 * 4u;
                let i0 = ic * stride + spatial_idx;
                let v = vec4f(input[i0], input[i0 + stride], input[i0 + stride * 2u], input[i0 + stride * 3u]);
                let w = vec4f(weights[w_base + done8 + ic4 * 4u], weights[w_base + done8 + ic4 * 4u + 1u], weights[w_base + done8 + ic4 * 4u + 2u], weights[w_base + done8 + ic4 * 4u + 3u]);
                s += dot(v, w);
            }
            let done4 = done8 + cpg4_rem * 4u;
            for (var ic = in_c_start + done4; ic < in_c_start + cpg; ic++) {
                s += input[ic * stride + spatial_idx] * weights[w_base + (ic - in_c_start)];
            }

            // Inline residual + activation + write
            if (params.has_residual == 1u) {
                s += residual[cur_oc * out_stride + oh * params.out_w + ow];
            }
            if (params.has_prelu == 1u) {
                if (s < 0.0) { s = s * prelu_slope[cur_oc]; }
            } else if (params.has_prelu == 2u) {
                s = clamp(s, 0.0, 6.0);
            } else if (params.has_prelu == 3u) {
                s = max(s, 0.0);
            }
            output[cur_oc * out_stride + oh * params.out_w + ow] = s;
        }
        return; // skip shared epilogue
    }
    // === GENERAL PATH ===
    else if (params.kern_h == 2u && params.kern_w == 2u) {
        // Unrolled 2x2 general conv (face landmark model's strided downsample convs)
        let ih0 = oh * params.stride_h - params.pad_top;
        let iw0 = ow * params.stride_w - params.pad_left;
        let ih1 = ih0 + 1u;
        let iw1 = iw0 + 1u;
        let sp = params.in_h * params.in_w;
        let cpg = channels_per_group;
        let ksize = cpg * 4u;
        for (var ic: u32 = 0u; ic < cpg; ic++) {
            let base = (in_c_start + ic) * sp;
            let wb = oc * ksize + ic * 4u;
            if (ih0 < params.in_h && iw0 < params.in_w) { sum += input[base + ih0 * params.in_w + iw0] * weights[wb]; }
            if (ih0 < params.in_h && iw1 < params.in_w) { sum += input[base + ih0 * params.in_w + iw1] * weights[wb + 1u]; }
            if (ih1 < params.in_h && iw0 < params.in_w) { sum += input[base + ih1 * params.in_w + iw0] * weights[wb + 2u]; }
            if (ih1 < params.in_h && iw1 < params.in_w) { sum += input[base + ih1 * params.in_w + iw1] * weights[wb + 3u]; }
        }
    }
    else {
        for (var ic: u32 = 0u; ic < channels_per_group; ic++) {
            let in_c_idx = in_c_start + ic;
            for (var kh: u32 = 0u; kh < params.kern_h; kh++) {
                for (var kw: u32 = 0u; kw < params.kern_w; kw++) {
                    let ih = oh * params.stride_h + kh - params.pad_top;
                    let iw = ow * params.stride_w + kw - params.pad_left;
                    if (ih < params.in_h && iw < params.in_w) {
                        let in_idx = in_c_idx * params.in_h * params.in_w + ih * params.in_w + iw;
                        let w_idx = oc * channels_per_group * params.kern_h * params.kern_w
                                  + ic * params.kern_h * params.kern_w
                                  + kh * params.kern_w + kw;
                        sum += input[in_idx] * weights[w_idx];
                    }
                }
            }
        }
    }

    // Shared epilogue for DW and general paths
    if (in_bounds) {
        if (params.has_residual == 1u) {
            let out_idx = oc * params.out_h * params.out_w + oh * params.out_w + ow;
            sum += residual[out_idx];
        }
        if (params.has_prelu == 1u) {
            if (sum < 0.0) { sum = sum * prelu_slope[oc]; }
        } else if (params.has_prelu == 2u) {
            sum = clamp(sum, 0.0, 6.0);
        } else if (params.has_prelu == 3u) {
            sum = max(sum, 0.0);
        }
        let out_idx = oc * params.out_h * params.out_w + oh * params.out_w + ow;
        output[out_idx] = sum;
    }
}
