// Fused inverted residual block: Expand 1x1 -> [ReLU6] -> DW 3x3 -> [ReLU6] -> Project 1x1 -> [Add]
// Single dispatch replaces 3-5 separate dispatches (+ activations).
// Designed for MobileNetV2 hand landmark model.
//
// Each thread computes one output pixel of the projection (narrowest point).
// The expand and DW results are computed on-the-fly per thread, never materialized.

struct InvResDesc {
    // Expand 1x1
    exp_in_ch: u32,        // input channels to expand
    exp_out_ch: u32,       // expanded channels (typically 6x input)
    exp_w_off: u32,        // offset into weights for expand kernel
    exp_b_off: u32,        // offset into weights for expand bias
    exp_act: u32,          // 0=none, 2=ReLU6, 3=ReLU

    // DW 3x3
    dw_stride: u32,
    dw_pad_t: u32,
    dw_pad_l: u32,
    dw_pad_b: u32,
    dw_pad_r: u32,
    dw_w_off: u32,         // offset for DW kernel
    dw_b_off: u32,         // offset for DW bias
    dw_act: u32,           // 0=none, 2=ReLU6, 3=ReLU

    // Project 1x1
    proj_out_ch: u32,      // output channels (narrow)
    proj_w_off: u32,       // offset for project kernel
    proj_b_off: u32,       // offset for project bias

    // Spatial dims
    in_h: u32,
    in_w: u32,
    out_h: u32,
    out_w: u32,

    // Residual
    has_residual: u32,     // 0=none, 1=add
}

@group(0) @binding(0) var<uniform> d: InvResDesc;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read> residual: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let ow = gid.x;
    let oh = gid.y;
    let oc = gid.z; // output channel of the projection

    if (ow >= d.out_w || oh >= d.out_h || oc >= d.proj_out_ch) { return; }

    // Start with project bias
    var proj_sum: f32 = weights[d.proj_b_off + oc];

    // For each expanded channel (DW is depthwise, so DW channel == expand channel)
    for (var ec: u32 = 0u; ec < d.exp_out_ch; ec++) {
        // --- Compute DW output at (oh, ow) for expand channel ec ---
        var dw_val: f32 = weights[d.dw_b_off + ec];

        for (var kh: u32 = 0u; kh < 3u; kh++) {
            for (var kw: u32 = 0u; kw < 3u; kw++) {
                let ih_padded = oh * d.dw_stride + kh;
                let iw_padded = ow * d.dw_stride + kw;
                let ih = ih_padded - d.dw_pad_t;
                let iw = iw_padded - d.dw_pad_l;

                // Bounds check (unsigned wrap handles negative)
                if (ih < d.in_h && iw < d.in_w) {
                    // --- Compute expand 1x1 output at (ih, iw) for channel ec ---
                    var exp_val: f32 = weights[d.exp_b_off + ec];
                    let spatial = ih * d.in_w + iw;

                    // Vectorized expand accumulation
                    let cpg4 = d.exp_in_ch / 4u;
                    for (var ic4: u32 = 0u; ic4 < cpg4; ic4++) {
                        let ic = ic4 * 4u;
                        let i0 = ic * d.in_h * d.in_w + spatial;
                        let s = d.in_h * d.in_w;
                        let v = vec4f(input[i0], input[i0+s], input[i0+s*2u], input[i0+s*3u]);
                        let w = vec4f(
                            weights[d.exp_w_off + ec * d.exp_in_ch + ic],
                            weights[d.exp_w_off + ec * d.exp_in_ch + ic + 1u],
                            weights[d.exp_w_off + ec * d.exp_in_ch + ic + 2u],
                            weights[d.exp_w_off + ec * d.exp_in_ch + ic + 3u]
                        );
                        exp_val += dot(v, w);
                    }
                    for (var ic = cpg4 * 4u; ic < d.exp_in_ch; ic++) {
                        exp_val += input[ic * d.in_h * d.in_w + spatial] * weights[d.exp_w_off + ec * d.exp_in_ch + ic];
                    }

                    // Expand activation
                    if (d.exp_act == 2u) { exp_val = clamp(exp_val, 0.0, 6.0); }
                    else if (d.exp_act == 3u) { exp_val = max(exp_val, 0.0); }

                    let dw_w_idx = d.dw_w_off + ec * 9u + kh * 3u + kw;
                    dw_val += exp_val * weights[dw_w_idx];
                }
            }
        }

        // DW activation
        if (d.dw_act == 2u) { dw_val = clamp(dw_val, 0.0, 6.0); }
        else if (d.dw_act == 3u) { dw_val = max(dw_val, 0.0); }

        // Accumulate into project output
        proj_sum += dw_val * weights[d.proj_w_off + oc * d.exp_out_ch + ec];
    }

    // Residual add
    if (d.has_residual == 1u) {
        let sp = oc * d.out_h * d.out_w + oh * d.out_w + ow;
        proj_sum += residual[sp];
    }

    output[oc * d.out_h * d.out_w + oh * d.out_w + ow] = proj_sum;
}
