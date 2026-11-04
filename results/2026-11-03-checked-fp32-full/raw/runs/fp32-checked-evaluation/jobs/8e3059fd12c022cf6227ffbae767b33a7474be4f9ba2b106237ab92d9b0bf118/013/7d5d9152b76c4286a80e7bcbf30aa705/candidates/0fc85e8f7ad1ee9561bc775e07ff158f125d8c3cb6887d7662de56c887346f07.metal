#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tgid [[threadgroup_position_in_grid]], uint3 tid3 [[thread_position_in_threadgroup]], uint3 tgs [[threads_per_threadgroup]]) {
    (void)epsilon;
    uint row = tgid.x;
    if (row >= B) { return; }
    uint tid = tid3.x;
    uint tg = tgs.x;
    uint base = row * D;
    threadgroup float buf[4096];
    threadgroup float red[512];
    float local_max = -3.402823466e+38f;
    for (uint i = tid; i < D; i += tg) { local_max = max(local_max, x[base + i]); }
    red[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg >> 1; stride > 0u; stride >>= 1) { if (tid < stride) { red[tid] = max(red[tid], red[tid + stride]); } threadgroup_barrier(mem_flags::mem_threadgroup); }
    float row_max = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_sum = 0.0f;
    if (D <= 4096u) {
        for (uint i = tid; i < D; i += tg) { float e = exp(x[base + i] - row_max); buf[i] = e; local_sum += e; }
    } else {
        for (uint i = tid; i < D; i += tg) { local_sum += exp(x[base + i] - row_max); }
    }
    red[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg >> 1; stride > 0u; stride >>= 1) { if (tid < stride) { red[tid] += red[tid + stride]; } threadgroup_barrier(mem_flags::mem_threadgroup); }
    float inv_sum = 1.0f / red[0];
    if (D <= 4096u) {
        for (uint i = tid; i < D; i += tg) { out[base + i] = float(buf[i] * inv_sum); }
    } else {
        for (uint i = tid; i < D; i += tg) { out[base + i] = float(exp(x[base + i] - row_max) * inv_sum); }
    }
}
