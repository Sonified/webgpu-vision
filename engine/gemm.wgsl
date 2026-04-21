// General Matrix Multiply (fully-connected layer).
// Computes: output = input * weights + bias
//
// Two paths (selected by uniform N, not thread ID):
//   N >= 16: one thread per output column, vec4-unrolled dot product over K,
//            input vector cached in workgroup shared memory.
//   N < 16:  parallel reduction -- all 64 threads split K for each output column,
//            then reduce via shared memory. Handles the N=1 case (672x1).

struct GemmParams {
    M: u32,
    K: u32,
    N: u32,
    has_bias: u32,
    has_sigmoid: u32,
}

@group(0) @binding(0) var<uniform> params: GemmParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

var<workgroup> shared_input: array<f32, 1024>;
var<workgroup> shared_reduce: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wgid: vec3u,
) {
    let tid = lid.x;
    let K = params.K;
    let N = params.N;

    // Cooperatively load input row 0 into shared memory
    let k_chunks = (K + 63u) / 64u;
    for (var c: u32 = 0u; c < k_chunks; c++) {
        let idx = c * 64u + tid;
        if (idx < K && idx < 1024u) {
            shared_input[idx] = input[idx];
        }
    }
    workgroupBarrier();

    if (N < 16u) {
        // === PARALLEL REDUCTION PATH (small N, e.g. 672x1) ===
        for (var col: u32 = 0u; col < N; col++) {
            let k_per_thread = (K + 63u) / 64u;
            let k_start = tid * k_per_thread;
            let k_end = min(k_start + k_per_thread, K);

            var partial: f32 = 0.0;

            let k_aligned = k_start + ((k_end - k_start) / 4u) * 4u;
            var k: u32 = k_start;
            for (; k < k_aligned; k += 4u) {
                let i = vec4f(shared_input[k], shared_input[k+1u], shared_input[k+2u], shared_input[k+3u]);
                let w_base = k * N + col;
                let w = vec4f(weights[w_base], weights[w_base + N], weights[w_base + N*2u], weights[w_base + N*3u]);
                partial += dot(i, w);
            }
            for (; k < k_end; k++) {
                partial += shared_input[k] * weights[k * N + col];
            }

            shared_reduce[tid] = partial;
            workgroupBarrier();

            if (tid < 32u) { shared_reduce[tid] += shared_reduce[tid + 32u]; }
            workgroupBarrier();
            if (tid < 16u) { shared_reduce[tid] += shared_reduce[tid + 16u]; }
            workgroupBarrier();
            if (tid < 8u) { shared_reduce[tid] += shared_reduce[tid + 8u]; }
            workgroupBarrier();
            if (tid < 4u) { shared_reduce[tid] += shared_reduce[tid + 4u]; }
            workgroupBarrier();
            if (tid < 2u) { shared_reduce[tid] += shared_reduce[tid + 2u]; }
            workgroupBarrier();
            if (tid == 0u) {
                var val = shared_reduce[0u] + shared_reduce[1u];
                if (params.has_bias == 1u) { val += bias[col]; }
                if (params.has_sigmoid == 1u) { val = 1.0 / (1.0 + exp(-val)); }
                output[col] = val;
            }
            workgroupBarrier();
        }
    } else {
        // === WIDE OUTPUT PATH (N >= 16, e.g. 672x63) ===
        let col = wgid.x * 64u + tid;
        if (col >= N) { return; }

        var sum: f32 = 0.0;
        if (params.has_bias == 1u) { sum = bias[col]; }

        let K4 = K / 4u;
        for (var k4: u32 = 0u; k4 < K4; k4++) {
            let k = k4 * 4u;
            let i = vec4f(shared_input[k], shared_input[k+1u], shared_input[k+2u], shared_input[k+3u]);
            let w_base = k * N + col;
            let w = vec4f(weights[w_base], weights[w_base + N], weights[w_base + N*2u], weights[w_base + N*3u]);
            sum += dot(i, w);
        }
        for (var k: u32 = K4 * 4u; k < K; k++) {
            sum += shared_input[k] * weights[k * N + col];
        }

        if (params.has_sigmoid == 1u) { sum = 1.0 / (1.0 + exp(-sum)); }
        output[col] = sum;
    }
}
