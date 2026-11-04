#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tgid [[threadgroup_position_in_grid]], uint3 tid_in_tg [[thread_position_in_threadgroup]], uint3 tg_size [[threads_per_threadgroup]]) {
    uint row = tgid.x;
    if (row >= B) { return; }
    uint tid = tid_in_tg.x;
    uint tg = tg_size.x;
    threadgroup float shared[256];
    threadgroup float inv_shared;
    float sum = 0.0f;
    uint base = row * D;
    for (uint col = tid; col < D; col += tg) {
        float v = x[base + col];
        sum += v * v;
    }
    if (tid < 256u) { shared[tid] = sum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) { shared[tid] += shared[tid + stride]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        inv_shared = rsqrt(shared[0] / float(D) + epsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = inv_shared;
    for (uint col = tid; col < D; col += tg) {
        float v = x[base + col];
        out[base + col] = float(v * inv * parameter[col]);
    }
}
