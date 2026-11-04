#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 group_pos [[threadgroup_position_in_grid]], uint3 thread_pos [[thread_position_in_threadgroup]], uint3 group_dims [[threads_per_threadgroup]]) {
    (void)epsilon;
    uint row = group_pos.x;
    if (row >= B || D == 0u) {
        return;
    }
    uint tid = thread_pos.x;
    uint tgs = group_dims.x;
    threadgroup float shared_data[256];
    uint base = row * D;
    float local_max = -3.402823466e38f;
    for (uint i = tid; i < D; i += tgs) {
        local_max = max(local_max, x[base + i]);
    }
    shared_data[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgs / 2u; stride > 0u; stride /= 2u) {
        if (tid < stride) {
            shared_data[tid] = max(shared_data[tid], shared_data[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared_data[0];
    float local_sum = 0.0f;
    for (uint i = tid; i < D; i += tgs) {
        local_sum += exp(x[base + i] - row_max);
    }
    shared_data[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgs / 2u; stride > 0u; stride /= 2u) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / shared_data[0];
    for (uint i = tid; i < D; i += tgs) {
        out[base + i] = exp(x[base + i] - row_max) * inv_sum;
    }
}
