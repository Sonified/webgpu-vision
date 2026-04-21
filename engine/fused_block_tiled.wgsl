// Tiled variant of fused_block for small spatial outputs (<=8x8).
// Shared memory tile per input channel eliminates redundant global reads.
// Manually unrolled 3x3 and 5x5 DW kernels.
// Uses separate pipeline from fused_block.wgsl to avoid occupancy penalty
// from shared memory allocation on large-spatial dispatches.

struct BlockDesc {
    dw_in_ch: u32,
    dw_kern: u32,
    dw_stride: u32,
    dw_pad_t: u32,
    dw_pad_l: u32,
    dw_pad_b: u32,
    dw_pad_r: u32,
    dw_act: u32,
    dw_w_off: u32,
    dw_b_off: u32,
    pw_out_ch: u32,
    pw_w_off: u32,
    pw_b_off: u32,
    in_h: u32,
    in_w: u32,
    out_h: u32,
    out_w: u32,
    has_residual: u32,
    res_ch: u32,
    act_type: u32,
    act_off: u32,
}

@group(0) @binding(0) var<uniform> d: BlockDesc;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read> residual: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

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

    let in_bounds = ow < d.out_w && oh < d.out_h && oc < d.pw_out_ch;

    var pw_sum: f32 = 0.0;
    if (in_bounds) { pw_sum = weights[d.pw_b_off + oc]; }

    let kern = d.dw_kern;
    let tile_w: u32 = 8u * d.dw_stride + kern - 1u;
    let origin_h: i32 = i32(wgid.y * 8u * d.dw_stride) - i32(d.dw_pad_t);
    let origin_w: i32 = i32(wgid.x * 8u * d.dw_stride) - i32(d.dw_pad_l);
    let local_idx = lid.y * 8u + lid.x;
    let tile_size = tile_w * (8u * d.dw_stride + kern - 1u);
    let sp = d.in_h * d.in_w;

    for (var ic: u32 = 0u; ic < d.dw_in_ch; ic++) {
        let ch_base = ic * sp;
        for (var t = local_idx; t < tile_size; t += 64u) {
            let ty = i32(t / tile_w);
            let tx = i32(t % tile_w);
            let iy = origin_h + ty;
            let ix = origin_w + tx;
            var val: f32 = 0.0;
            if (iy >= 0 && iy < i32(d.in_h) && ix >= 0 && ix < i32(d.in_w)) {
                val = input[ch_base + u32(iy) * d.in_w + u32(ix)];
            }
            tile[t] = val;
        }
        workgroupBarrier();

        if (in_bounds) {
            var dw_val: f32 = weights[d.dw_b_off + ic];
            let lo = lid.y * d.dw_stride;
            let lw = lid.x * d.dw_stride;
            let tw = tile_w;
            let wb = d.dw_w_off + ic * kern * kern;

            if (kern == 3u) {
                let b = lo * tw + lw;
                dw_val += tile[b]           * weights[wb];
                dw_val += tile[b + 1u]      * weights[wb + 1u];
                dw_val += tile[b + 2u]      * weights[wb + 2u];
                dw_val += tile[b + tw]      * weights[wb + 3u];
                dw_val += tile[b + tw + 1u] * weights[wb + 4u];
                dw_val += tile[b + tw + 2u] * weights[wb + 5u];
                let b2 = b + tw * 2u;
                dw_val += tile[b2]          * weights[wb + 6u];
                dw_val += tile[b2 + 1u]     * weights[wb + 7u];
                dw_val += tile[b2 + 2u]     * weights[wb + 8u];
            } else if (kern == 5u) {
                let b = lo * tw + lw;
                dw_val += tile[b]           * weights[wb];
                dw_val += tile[b + 1u]      * weights[wb + 1u];
                dw_val += tile[b + 2u]      * weights[wb + 2u];
                dw_val += tile[b + 3u]      * weights[wb + 3u];
                dw_val += tile[b + 4u]      * weights[wb + 4u];
                let r1 = b + tw;
                dw_val += tile[r1]          * weights[wb + 5u];
                dw_val += tile[r1 + 1u]     * weights[wb + 6u];
                dw_val += tile[r1 + 2u]     * weights[wb + 7u];
                dw_val += tile[r1 + 3u]     * weights[wb + 8u];
                dw_val += tile[r1 + 4u]     * weights[wb + 9u];
                let r2 = b + tw * 2u;
                dw_val += tile[r2]          * weights[wb + 10u];
                dw_val += tile[r2 + 1u]     * weights[wb + 11u];
                dw_val += tile[r2 + 2u]     * weights[wb + 12u];
                dw_val += tile[r2 + 3u]     * weights[wb + 13u];
                dw_val += tile[r2 + 4u]     * weights[wb + 14u];
                let r3 = b + tw * 3u;
                dw_val += tile[r3]          * weights[wb + 15u];
                dw_val += tile[r3 + 1u]     * weights[wb + 16u];
                dw_val += tile[r3 + 2u]     * weights[wb + 17u];
                dw_val += tile[r3 + 3u]     * weights[wb + 18u];
                dw_val += tile[r3 + 4u]     * weights[wb + 19u];
                let r4 = b + tw * 4u;
                dw_val += tile[r4]          * weights[wb + 20u];
                dw_val += tile[r4 + 1u]     * weights[wb + 21u];
                dw_val += tile[r4 + 2u]     * weights[wb + 22u];
                dw_val += tile[r4 + 3u]     * weights[wb + 23u];
                dw_val += tile[r4 + 4u]     * weights[wb + 24u];
            } else {
                for (var kh: u32 = 0u; kh < kern; kh++) {
                    for (var kw: u32 = 0u; kw < kern; kw++) {
                        dw_val += tile[(lo + kh) * tw + (lw + kw)] * weights[wb + kh * kern + kw];
                    }
                }
            }

            if (d.dw_act == 2u) { dw_val = clamp(dw_val, 0.0, 6.0); }
            else if (d.dw_act == 3u) { dw_val = max(dw_val, 0.0); }

            pw_sum += dw_val * weights[d.pw_w_off + oc * d.dw_in_ch + ic];
        }
        workgroupBarrier();
    }

    if (in_bounds) {
        if (d.has_residual >= 1u) {
            let sp_out = oh * d.out_w + ow;
            if (d.has_residual == 2u) {
                if (oc < d.res_ch) { pw_sum += residual[oc * d.out_h * d.out_w + sp_out]; }
            } else {
                pw_sum += residual[oc * d.out_h * d.out_w + sp_out];
            }
        }
        if (d.act_type == 1u) {
            if (pw_sum < 0.0) { pw_sum *= weights[d.act_off + oc]; }
        } else if (d.act_type == 2u) {
            pw_sum = clamp(pw_sum, 0.0, 6.0);
        } else if (d.act_type == 3u) {
            pw_sum = max(pw_sum, 0.0);
        }
        output[oc * d.out_h * d.out_w + oh * d.out_w + ow] = pw_sum;
    }
}
