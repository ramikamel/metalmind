#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint thread_position_in_threadgroup [[thread_position_in_threadgroup]], uint threadgroup_position_in_grid [[threadgroup_position_in_grid]], uint threads_per_threadgroup [[threads_per_threadgroup]]) {
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint tg = threads_per_threadgroup.x;
    threadgroup float shared[256];
    float sum = 0.0f;
    if (row < B && D > 0u) {
        uint base = row * D;
        for (uint j = tid; j < D; j += tg) {
            float v = x[base + j];
            sum += v * v;
        }
    }
    if (tid < 256u) {
        shared[tid] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float scale = 0.0f;
    if (D > 0u) {
        scale = rsqrt(shared[0] / float(D) + epsilon);
    }
    if (row < B && D > 0u) {
        uint base = row * D;
        for (uint j = tid; j < D; j += tg) {
            float v = x[base + j];
            out[base + j] = float(v * scale * parameter[j]);
        }
    }
}
