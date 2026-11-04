#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tgid [[threadgroup_position_in_grid]], uint3 tid_in_group [[thread_position_in_threadgroup]], uint3 tg_size [[threads_per_threadgroup]]) {
    uint row = tgid.x;
    if (row >= B) return;
    uint tid = tid_in_group.x;
    uint tgs = tg_size.x;
    threadgroup float partial[256];
    threadgroup float shared_inv;
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
    if (tid == 0) {
        float mean = partial[0] / float(D);
        shared_inv = rsqrt(mean + epsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = shared_inv;
    for (uint i = tid; i < D; i += tgs) {
        uint idx = base + i;
        float v = x[idx];
        out[idx] = v * inv * parameter[i];
    }
}
