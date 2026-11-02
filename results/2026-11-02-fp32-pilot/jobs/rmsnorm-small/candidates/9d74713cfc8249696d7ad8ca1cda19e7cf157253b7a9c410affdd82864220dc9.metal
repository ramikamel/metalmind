#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint tgid [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tgdim [[threads_per_threadgroup]]) {
    uint row = tgid;
    if (row >= B) return;
    uint base = row * D;
    float local = 0.0f;
    for (uint i = tid; i < D; i += tgdim) {
        float v = x[base + i];
        local += v * v;
    }
    threadgroup float shared[256];
    shared[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgdim >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(shared[0] / float(D) + epsilon);
    for (uint i = tid; i < D; i += tgdim) {
        uint idx = base + i;
        out[idx] = x[idx] * inv * parameter[i];
    }
}
