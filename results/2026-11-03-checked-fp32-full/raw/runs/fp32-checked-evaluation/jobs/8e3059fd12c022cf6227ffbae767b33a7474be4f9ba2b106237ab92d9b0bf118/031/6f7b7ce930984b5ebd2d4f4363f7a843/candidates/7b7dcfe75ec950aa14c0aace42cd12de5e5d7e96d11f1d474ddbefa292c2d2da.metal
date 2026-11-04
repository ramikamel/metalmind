#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tg_pos [[threadgroup_position_in_grid]], uint3 tid_in_tg [[thread_position_in_threadgroup]], uint3 threads_per_tg [[threads_per_threadgroup]]) {
    (void)epsilon;
    uint row = tg_pos.x;
    if (row >= B) { return; }
    uint tid = tid_in_tg.x;
    uint tpg = threads_per_tg.x;
    threadgroup float scratch[128];
    uint base = row * D;
    float local_max = -INFINITY;
    for (uint i = tid; i < D; i += tpg) { local_max = max(local_max, x[base + i]); }
    scratch[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tpg / 2; stride > 0; stride >>= 1) {
        if (tid < stride) { scratch[tid] = max(scratch[tid], scratch[tid + stride]); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = scratch[0];
    float local_sum = 0.0f;
    for (uint i = tid; i < D; i += tpg) { local_sum += exp(x[base + i] - row_max); }
    scratch[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tpg / 2; stride > 0; stride >>= 1) {
        if (tid < stride) { scratch[tid] += scratch[tid + stride]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_sum = scratch[0];
    float inv_sum = 1.0f / row_sum;
    for (uint i = tid; i < D; i += tpg) { out[base + i] = exp(x[base + i] - row_max) * inv_sum; }
}
