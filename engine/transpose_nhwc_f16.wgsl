enable f16;

struct TransposeParams {
    channels: u32,
    height: u32,
    width: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: TransposeParams;
@group(0) @binding(1) var<storage, read> input: array<f16>;
@group(0) @binding(2) var<storage, read_write> output: array<f16>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    let total = params.channels * params.height * params.width;
    if (idx >= total) { return; }

    let c = idx % params.channels;
    let w = (idx / params.channels) % params.width;
    let h = idx / (params.channels * params.width);

    let nchw_idx = c * params.height * params.width + h * params.width + w;

    output[idx] = input[nchw_idx];
}
