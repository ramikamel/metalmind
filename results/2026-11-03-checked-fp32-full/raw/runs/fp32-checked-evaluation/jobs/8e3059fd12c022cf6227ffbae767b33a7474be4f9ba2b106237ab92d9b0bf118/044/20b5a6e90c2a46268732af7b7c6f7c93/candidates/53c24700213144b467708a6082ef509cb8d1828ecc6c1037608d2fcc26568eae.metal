#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint row [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]], uint tpg [[threads_per_threadgroup]]) {
    (void)epsilon;
    if (row >= B) return;
    threadgroup float shared[256];
    uint base = row * D;
    float local_max = -3.4028235e38f;
    for (uint i = lid; i < D; i += tpg) {
        local_max = max(local_max, x[base + i]);
    }
    shared[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tpg >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) shared[lid] = max(shared[lid], shared[lid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_sum = 0.0f;
    for (uint i = lid; i < D; i += tpg) {
        local_sum += exp(x[base + i] - row_max);
    }
    shared[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tpg >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) shared[lid] += shared[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_sum = shared[0];
    float inv_sum = 1.0f / row_sum;
    for (uint i = lid; i < D; i += tpg) {
        out[base + i] = float(exp(x[base + i] - row_max) * inv_sum);
    }
}
