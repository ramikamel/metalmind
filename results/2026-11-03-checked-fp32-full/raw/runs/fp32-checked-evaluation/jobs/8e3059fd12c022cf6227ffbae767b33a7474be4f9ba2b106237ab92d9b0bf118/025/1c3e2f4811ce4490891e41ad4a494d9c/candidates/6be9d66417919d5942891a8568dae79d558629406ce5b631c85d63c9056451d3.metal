#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint tpg [[threads_per_threadgroup]]) {
    if (tg >= B) { return; }
    threadgroup float partial[256];
    uint base = tg * D;
    float local = 0.0f;
    for (uint i = tid; i < D; i += tpg) {
        float v = x[base + i];
        local += v * v;
    }
    partial[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tpg / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float scale = rsqrt(partial[0] / float(D) + epsilon);
    for (uint i = tid; i < D; i += tpg) {
        float v = x[base + i];
        out[base + i] = v * scale * parameter[i];
    }
}
