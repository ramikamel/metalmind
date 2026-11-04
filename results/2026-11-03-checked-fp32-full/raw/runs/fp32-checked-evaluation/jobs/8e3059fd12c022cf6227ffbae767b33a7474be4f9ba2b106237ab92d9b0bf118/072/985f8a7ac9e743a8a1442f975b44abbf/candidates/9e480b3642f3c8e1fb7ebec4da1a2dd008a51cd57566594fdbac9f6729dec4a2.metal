#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tg_size [[threads_per_threadgroup]]) {
    threadgroup float partial[256];
    if (row >= B) {
        return;
    }
    float sum = 0.0f;
    uint base = row * D;
    for (uint j = tid; j < D; j += tg_size) {
        float v = x[base + j];
        sum += v * v;
    }
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float mean = partial[0] / float(D);
        partial[0] = rsqrt(mean + epsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = partial[0];
    for (uint j = tid; j < D; j += tg_size) {
        float v = x[base + j];
        out[base + j] = float(v * scale * parameter[j]);
    }
}
