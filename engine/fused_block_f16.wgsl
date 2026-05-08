enable f16;

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
@group(0) @binding(1) var<storage, read> input: array<f16>;
@group(0) @binding(2) var<storage, read> weights: array<f16>;
@group(0) @binding(3) var<storage, read> residual: array<f16>;
@group(0) @binding(4) var<storage, read_write> output: array<f16>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let ow = gid.x;
    let oh = gid.y;
    let oc = gid.z;

    if (ow >= d.out_w || oh >= d.out_h || oc >= d.pw_out_ch) { return; }

    var pw_sum: f16 = weights[d.pw_b_off + oc];

    let kern = d.dw_kern;
    for (var ic: u32 = 0u; ic < d.dw_in_ch; ic++) {
        var dw_val: f16 = weights[d.dw_b_off + ic];
        for (var kh: u32 = 0u; kh < kern; kh++) {
            for (var kw: u32 = 0u; kw < kern; kw++) {
                let ih_padded = oh * d.dw_stride + kh;
                let iw_padded = ow * d.dw_stride + kw;
                let ih = ih_padded - d.dw_pad_t;
                let iw = iw_padded - d.dw_pad_l;
                if (ih < d.in_h && iw < d.in_w) {
                    let in_idx = ic * d.in_h * d.in_w + ih * d.in_w + iw;
                    let w_idx = d.dw_w_off + ic * kern * kern + kh * kern + kw;
                    dw_val += input[in_idx] * weights[w_idx];
                }
            }
        }

        if (d.dw_act == 2u) {
            dw_val = clamp(dw_val, 0.0h, 6.0h);
        } else if (d.dw_act == 3u) {
            dw_val = max(dw_val, 0.0h);
        }

        let pw_w_idx = d.pw_w_off + oc * d.dw_in_ch + ic;
        pw_sum += dw_val * weights[pw_w_idx];
    }

    if (d.has_residual >= 1u) {
        let sp = oh * d.out_w + ow;
        if (d.has_residual == 2u) {
            if (oc < d.res_ch) {
                pw_sum += residual[oc * d.out_h * d.out_w + sp];
            }
        } else {
            pw_sum += residual[oc * d.out_h * d.out_w + sp];
        }
    }

    if (d.act_type == 1u) {
        if (pw_sum < 0.0h) { pw_sum *= weights[d.act_off + oc]; }
    } else if (d.act_type == 2u) {
        pw_sum = clamp(pw_sum, 0.0h, 6.0h);
    } else if (d.act_type == 3u) {
        pw_sum = max(pw_sum, 0.0h);
    }

    output[oc * d.out_h * d.out_w + oh * d.out_w + ow] = pw_sum;
}
