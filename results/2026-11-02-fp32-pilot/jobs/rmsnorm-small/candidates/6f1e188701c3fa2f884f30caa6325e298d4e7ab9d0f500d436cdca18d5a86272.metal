#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint2 tgid [[threadgroup_position_in_grid]], uint2 tid [[thread_position_in_threadgroup]], uint2 tgdim [[threads_per_threadgroup]]) {
    uint row = tgid.y;
    if (row >= B) return;
    uint base = row * D;
    uint lane = tid.x;
    uint width = tgdim.x;
    float local = 0.0f;
    for (uint i = lane; i < D; i += width) {
        float v = x[base + i];
        local += v * v;
    }
    float total = simd_sum(local);
    float inv = rsqrt(total / float(D) + epsilon);
    for (uint i = lane; i < D; i += width) {
        uint idx = base + i;
        out[idx] = x[idx] * inv * parameter[i];
    }
}
