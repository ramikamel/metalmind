#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 gid [[thread_position_in_grid]], uint3 tid3 [[thread_position_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
    uint row = tg.x;
    if (row >= B) return;
    uint tid = tid3.x;
    threadgroup float partial[256];
    threadgroup float scale;
    uint base = row * D;
    float sum = 0.0f;
    for (uint i = tid; i < D; i += 256u) {
        float v = x[base + i];
        sum += v * v;
    }
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) partial[tid] += partial[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) scale = rsqrt(partial[0] / float(D) + epsilon);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sc = scale;
    for (uint i = tid; i < D; i += 256u) {
        out[base + i] = x[base + i] * sc * parameter[i];
    }
}
