enable f16;
alias acc = f32; // accumulator precision -- change to f16 for pure f16 accumulation

struct PoolParams {
    channels: u32,
    height: u32,
    width: u32,
}

@group(0) @binding(0) var<uniform> params: PoolParams;
@group(0) @binding(1) var<storage, read> input: array<f16>;
@group(0) @binding(2) var<storage, read_write> output: array<f16>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let c = gid.x;
    if (c >= params.channels) { return; }

    let spatial = params.height * params.width;
    let base = c * spatial;
    var sum: acc = 0.0;
    for (var i: u32 = 0u; i < spatial; i++) {
        sum += acc(input[base + i]);
    }
    output[c] = f16(sum / acc(spatial));
}
