// Channel padding: copies input and appends zero channels.
// Input: [1, C_in, H, W], Output: [1, C_out, H, W] where C_out > C_in.
// Channels C_in..C_out-1 are filled with zeros.

struct PadParams {
    in_channels: u32,
    out_channels: u32,
    height: u32,
    width: u32,
}

@group(0) @binding(0) var<uniform> params: PadParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let w = gid.x;
    let h = gid.y;
    let c = gid.z;

    if (w >= params.width || h >= params.height || c >= params.out_channels) { return; }

    let idx = c * params.height * params.width + h * params.width + w;

    if (c < params.in_channels) {
        output[idx] = input[c * params.height * params.width + h * params.width + w];
    } else {
        output[idx] = 0.0;
    }
}
