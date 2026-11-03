#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint tg_size [[threads_per_threadgroup]]) {
    if (row >= B || D == 0u) {
        return;
    }
    threadgroup float partial[256];
    uint base = row * D;
    float sum = 0.0f;
    for (uint j = tid; j < D; j += tg_size) {
        float v = x[base + j];
        sum += v * v;
    }
    if (tid < 256u) {
        partial[tid] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = partial[0] / float(D);
    float inv = rsqrt(mean + epsilon);
    for (uint j = tid; j < D; j += tg_size) {
        float v = x[base + j];
        out[base + j] = v * inv * parameter[j];
    }
}
