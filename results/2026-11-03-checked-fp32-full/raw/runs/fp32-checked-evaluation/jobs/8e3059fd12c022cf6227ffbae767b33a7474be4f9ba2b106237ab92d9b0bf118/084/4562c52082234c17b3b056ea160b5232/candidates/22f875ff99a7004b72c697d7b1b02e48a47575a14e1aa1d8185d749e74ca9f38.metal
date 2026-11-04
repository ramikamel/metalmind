#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tg_pos [[threadgroup_position_in_grid]], uint3 tid_in_tg [[thread_position_in_threadgroup]], uint3 threads_per_tg [[threads_per_threadgroup]]) {
    uint row = tg_pos.x;
    if (row >= B) return;
    uint lid = tid_in_tg.x;
    uint tgs = threads_per_tg.x;
    if (tgs > 256u) return;
    threadgroup float shared[256];
    uint base = row * D;
    float local_max = -3.402823466e+38f;
    for (uint i = lid; i < D; i += tgs) {
        local_max = max(local_max, x[base + i]);
    }
    shared[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgs >> 1; stride > 0u; stride >>= 1) {
        if (lid < stride) shared[lid] = max(shared[lid], shared[lid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_sum = 0.0f;
    for (uint i = lid; i < D; i += tgs) {
        local_sum += exp(x[base + i] - row_max);
    }
    shared[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgs >> 1; stride > 0u; stride >>= 1) {
        if (lid < stride) shared[lid] += shared[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_sum = shared[0];
    float inv_sum = 1.0f / row_sum;
    for (uint i = lid; i < D; i += tgs) {
        float v = exp(x[base + i] - row_max) * inv_sum;
        out[base + i] = v;
    }
}
