#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 gid [[thread_position_in_grid]], uint3 tid3 [[thread_position_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]], uint3 tpg [[threads_per_threadgroup]]) {
    (void)gid;
    (void)epsilon;
    uint row = tg.x;
    if (row >= B) { return; }
    uint tid = tid3.x;
    uint nt = tpg.x;
    threadgroup float partial[256];
    threadgroup float shared_val;
    uint base = row * D;
    float local_max = -3.4028235e38f;
    for (uint i = tid; i < D; i += nt) {
        float v = float(x[base + i]);
        local_max = max(local_max, v);
    }
    partial[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float m = partial[0];
        for (uint i = 1; i < nt; ++i) { m = max(m, partial[i]); }
        shared_val = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = shared_val;
    float local_sum = 0.0f;
    for (uint i = tid; i < D; i += nt) {
        float v = float(x[base + i]);
        local_sum += exp(v - row_max);
    }
    partial[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = partial[0];
        for (uint i = 1; i < nt; ++i) { s += partial[i]; }
        shared_val = (s > 0.0f) ? (1.0f / s) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = shared_val;
    for (uint i = tid; i < D; i += nt) {
        float v = float(x[base + i]);
        out[base + i] = half(exp(v - row_max) * inv_sum);
    }
}
