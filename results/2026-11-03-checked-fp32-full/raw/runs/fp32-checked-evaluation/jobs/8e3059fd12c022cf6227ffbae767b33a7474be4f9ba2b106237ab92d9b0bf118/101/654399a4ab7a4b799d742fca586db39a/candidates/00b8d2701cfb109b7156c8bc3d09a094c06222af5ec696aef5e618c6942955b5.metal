#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint tid [[thread_position_in_threadgroup]], uint tgid [[threadgroup_position_in_grid]], uint tgdim [[threads_per_threadgroup]]) {
    (void)epsilon;
    threadgroup float shared[128];
    uint row = tgid;
    if (row >= B) {
        return;
    }
    uint base = row * D;
    float local_max = -1.0e30f;
    for (uint i = tid; i < D; i += tgdim) {
        local_max = max(local_max, x[base + i]);
    }
    shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgdim / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] = max(shared[tid], shared[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_sum = 0.0f;
    for (uint i = tid; i < D; i += tgdim) {
        local_sum += exp(x[base + i] - row_max);
    }
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgdim / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_sum = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = 1.0f / row_sum;
    for (uint i = tid; i < D; i += tgdim) {
        out[base + i] = exp(x[base + i] - row_max) * inv_sum;
    }
}
