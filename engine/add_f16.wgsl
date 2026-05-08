enable f16;

struct AddParams {
    count: u32,
    mode: u32,
    channels: u32,
    spatial: u32,
}

@group(0) @binding(0) var<uniform> params: AddParams;
@group(0) @binding(1) var<storage, read> a: array<f16>;
@group(0) @binding(2) var<storage, read> b: array<f16>;
@group(0) @binding(3) var<storage, read_write> output: array<f16>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.count) { return; }
    var val: f16;
    if (params.mode == 1u) {
        val = max(a[i], 0.0h);
    } else if (params.mode == 3u) {
        let ch = i / params.spatial;
        val = a[i];
        if (val < 0.0h) { val = val * b[ch]; }
    } else if (params.mode == 4u) {
        val = 1.0h / (1.0h + exp(-a[i]));
    } else if (params.mode == 5u) {
        val = round(a[i]);
    } else {
        val = a[i] + b[i];
        if (params.mode == 2u) { val = max(val, 0.0h); }
    }
    output[i] = val;
}
