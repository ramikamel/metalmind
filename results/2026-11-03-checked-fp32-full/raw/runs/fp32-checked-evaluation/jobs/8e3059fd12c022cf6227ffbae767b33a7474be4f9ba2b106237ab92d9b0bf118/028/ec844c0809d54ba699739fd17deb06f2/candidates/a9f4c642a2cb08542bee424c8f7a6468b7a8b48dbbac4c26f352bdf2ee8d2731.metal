#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint lid [[thread_position_in_threadgroup]], uint row [[threadgroup_position_in_grid]]) {
    if (row >= B) {
        return;
    }
    threadgroup float shared[256];
    uint base = row * D;
    float sum = 0.0f;
    for (uint c = lid; c < D; c += 256u) {
        float v = x[base + c];
        sum += v * v;
    }
    shared[lid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (lid < stride) {
            shared[lid] += shared[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(shared[0] / float(D) + epsilon);
    for (uint c = lid; c < D; c += 256u) {
        out[base + c] = x[base + c] * inv * parameter[c];
    }
}
