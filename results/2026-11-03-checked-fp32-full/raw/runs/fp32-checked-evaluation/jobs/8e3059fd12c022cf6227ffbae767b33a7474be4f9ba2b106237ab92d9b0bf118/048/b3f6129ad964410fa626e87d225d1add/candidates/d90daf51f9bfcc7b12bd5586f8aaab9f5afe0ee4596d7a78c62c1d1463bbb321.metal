#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]], uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]], uint3 threads_per_threadgroup [[threads_per_threadgroup]]) {
    uint row = threadgroup_position_in_grid.x;
    if (row >= B) return;
    uint tid = thread_position_in_threadgroup.x;
    uint tgs = threads_per_threadgroup.x;
    threadgroup float partial[1024];
    float sum = 0.0f;
    uint base = row * D;
    for (uint i = tid; i < D; i += tgs) {
        float v = x[base + i];
        sum += v * v;
    }
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgs >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(partial[0] / float(D) + epsilon);
    for (uint i = tid; i < D; i += tgs) {
        float v = x[base + i];
        out[base + i] = float(v * inv * parameter[i]);
    }
}
