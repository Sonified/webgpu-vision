enable f16;

struct PoolParams {
    channels: u32,
    in_h: u32,
    in_w: u32,
    out_h: u32,
    out_w: u32,
    out_channels: u32,
}

@group(0) @binding(0) var<uniform> params: PoolParams;
@group(0) @binding(1) var<storage, read> input: array<f16>;
@group(0) @binding(2) var<storage, read_write> output: array<f16>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let ow = gid.x;
    let oh = gid.y;
    let oc = gid.z;

    if (ow >= params.out_w || oh >= params.out_h || oc >= params.out_channels) {
        return;
    }

    let out_idx = oc * params.out_h * params.out_w + oh * params.out_w + ow;

    if (oc >= params.channels) {
        output[out_idx] = 0.0h;
        return;
    }

    let ih = oh * 2u;
    let iw = ow * 2u;
    let base = oc * params.in_h * params.in_w;

    var maxval: f16 = input[base + ih * params.in_w + iw];
    maxval = max(maxval, input[base + ih * params.in_w + iw + 1u]);
    maxval = max(maxval, input[base + (ih + 1u) * params.in_w + iw]);
    maxval = max(maxval, input[base + (ih + 1u) * params.in_w + iw + 1u]);

    output[out_idx] = maxval;
}
