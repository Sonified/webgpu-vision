// Element-wise ops: add, relu, prelu, or combinations.
// mode 0 = a + b
// mode 1 = relu(a) [b ignored]
// mode 2 = relu(a + b)
// mode 3 = prelu(a) using b as per-channel slopes [NCHW layout]

struct AddParams {
    count: u32,
    mode: u32,
    channels: u32,  // for prelu: number of channels (b.length)
    spatial: u32,   // for prelu: H * W (elements per channel)
}

@group(0) @binding(0) var<uniform> params: AddParams;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.count) { return; }
    var val: f32;
    if (params.mode == 1u) {
        val = max(a[i], 0.0);
    } else if (params.mode == 3u) {
        // PReLU: slope = b[channel], channel = i / spatial
        let ch = i / params.spatial;
        val = a[i];
        if (val < 0.0) { val = val * b[ch]; }
    } else {
        val = a[i] + b[i];
        if (params.mode == 2u) { val = max(val, 0.0); }
    }
    output[i] = val;
}
