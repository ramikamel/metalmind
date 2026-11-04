#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tgid [[threadgroup_position_in_grid]], uint3 tid3 [[thread_position_in_threadgroup]], uint3 nt3 [[threads_per_threadgroup]]) {
    uint row = tgid.x;
    if (row >= B) { return; }
    uint tid = tid3.x;
    uint nt = nt3.x;
    threadgroup float partial[256];
    threadgroup float invShared;
    uint base = row * D;
    float sum = 0.0f;
    for (uint c = tid; c < D; c += nt) {
        float v = x[base + c];
        sum += v * v;
    }
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) { partial[tid] += partial[tid + s]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        invShared = rsqrt(partial[0] / float(D) + epsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = invShared;
    for (uint c = tid; c < D; c += nt) {
        out[base + c] = x[base + c] * inv * parameter[c];
    }
}
