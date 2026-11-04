#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tgid [[threadgroup_position_in_grid]], uint3 tid_in_tg [[thread_position_in_threadgroup]], uint3 tg_size [[threads_per_threadgroup]]) {
    uint row = tgid.x;
    if (row >= B) return;
    uint tid = tid_in_tg.x;
    uint tg = tg_size.x;
    threadgroup float red[128];
    uint base = row * D;
    float sum = 0.0f;
    for (uint i = tid; i < D; i += tg) {
        float v = x[base + i];
        sum += v * v;
    }
    red[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(red[0] / float(D) + epsilon);
    for (uint i = tid; i < D; i += tg) {
        out[base + i] = x[base + i] * inv * parameter[i];
    }
}
