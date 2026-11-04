#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint tgsize [[threads_per_threadgroup]]) {
    if (row >= B) return;
    threadgroup float partial[256];
    threadgroup float inv_rms;
    uint base = row * D;
    float sum = 0.0f;
    for (uint i = tid; i < D; i += tgsize) {
        float v = x[base + i];
        sum += v * v;
    }
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgsize >> 1; s > 0; s >>= 1) {
        if (tid < s) partial[tid] += partial[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) inv_rms = rsqrt(partial[0] / float(D) + epsilon);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = inv_rms;
    for (uint i = tid; i < D; i += tgsize) {
        out[base + i] = x[base + i] * scale * parameter[i];
    }
}
