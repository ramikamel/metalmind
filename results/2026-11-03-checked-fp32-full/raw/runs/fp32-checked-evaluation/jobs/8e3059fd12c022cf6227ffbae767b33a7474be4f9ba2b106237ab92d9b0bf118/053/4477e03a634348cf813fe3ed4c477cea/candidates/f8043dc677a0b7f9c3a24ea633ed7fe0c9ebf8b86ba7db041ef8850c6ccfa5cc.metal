#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tg [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 tgs [[threads_per_threadgroup]]) {
    uint row = tg.x;
    if (row >= B) return;
    uint lane = tid.x;
    uint nthreads = tgs.x;
    uint base = row * D;
    float sum = 0.0f;
    for (uint i = lane; i < D; i += nthreads) {
        float v = x[base + i];
        sum += v * v;
    }
    threadgroup float partial[256];
    partial[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = nthreads >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float scale = rsqrt(partial[0] / float(D) + epsilon);
    for (uint i = lane; i < D; i += nthreads) {
        out[base + i] = float(x[base + i] * scale * parameter[i]);
    }
}
