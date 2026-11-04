#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]], uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]) {
    uint row = threadgroup_position_in_grid.x;
    if (row >= B) return;
    uint lid = thread_position_in_threadgroup.x;
    uint base = row * D;
    threadgroup float partial[128];
    float sum = 0.0f;
    for (uint i = lid; i < D; i += 128u) {
        float v = x[base + i];
        sum += v * v;
    }
    partial[lid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 64u; stride > 0u; stride >>= 1u) {
        if (lid < stride) partial[lid] += partial[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(partial[0] / float(D) + epsilon);
    for (uint i = lid; i < D; i += 128u) {
        out[base + i] = x[base + i] * inv * parameter[i];
    }
}
