#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint tgs [[threads_per_threadgroup]]) {
    (void)epsilon;
    if (tg >= B || D == 0) return;
    uint row = tg;
    uint base = row * D;
    threadgroup float red[256];
    float local_max = -3.402823466e+38f;
    for (uint i = tid; i < D; i += tgs) {
        float v = x[base + i];
        local_max = max(local_max, v);
    }
    if (tid < 256) red[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgs >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) red[tid] = max(red[tid], red[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_sum = 0.0f;
    for (uint i = tid; i < D; i += tgs) {
        float v = x[base + i];
        local_sum += exp(v - row_max);
    }
    if (tid < 256) red[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgs >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) red[tid] += red[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_sum = red[0];
    float inv_sum = 1.0f / row_sum;
    for (uint i = tid; i < D; i += tgs) {
        float v = x[base + i];
        float e = exp(v - row_max);
        out[base + i] = float(e * inv_sum);
    }
}
