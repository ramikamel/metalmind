#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 gid [[thread_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 tg [[threads_per_threadgroup]], uint3 grp [[threadgroup_position_in_grid]]) {
    uint row = grp.x;
    if (row >= B) return;
    threadgroup float partial[256];
    uint base = row * D;
    float sum = 0.0f;
    for (uint col = tid.x; col < D; col += tg.x) {
        float v = x[base + col];
        sum += v * v;
    }
    partial[tid.x] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg.x / 2; s > 0; s >>= 1) {
        if (tid.x < s) partial[tid.x] += partial[tid.x + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(partial[0] / float(D) + epsilon);
    for (uint col = tid.x; col < D; col += tg.x) {
        out[base + col] = x[base + col] * inv * parameter[col];
    }
}
