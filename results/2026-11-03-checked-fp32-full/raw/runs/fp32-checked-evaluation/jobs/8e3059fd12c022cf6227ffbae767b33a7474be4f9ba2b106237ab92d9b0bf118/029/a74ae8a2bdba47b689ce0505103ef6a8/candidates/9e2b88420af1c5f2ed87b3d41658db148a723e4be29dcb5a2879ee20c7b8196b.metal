#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]], uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]], uint3 threads_per_threadgroup [[threads_per_threadgroup]]) {
    uint row = threadgroup_position_in_grid.x;
    if (row >= B) {
        return;
    }
    uint tid = thread_position_in_threadgroup.x;
    uint nt = threads_per_threadgroup.x;
    threadgroup float shared_data[256];
    threadgroup float broadcast_val;
    device const float* row_ptr = x + ((uint64_t)row * (uint64_t)D);
    device float* out_ptr = out + ((uint64_t)row * (uint64_t)D);
    float local_max = -3.402823466e+38f;
    for (uint i = tid; i < D; i += nt) {
        local_max = max(local_max, row_ptr[i]);
    }
    shared_data[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = nt / 2u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_data[tid] = max(shared_data[tid], shared_data[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        broadcast_val = shared_data[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = broadcast_val;
    float local_sum = 0.0f;
    for (uint i = tid; i < D; i += nt) {
        local_sum += exp(row_ptr[i] - row_max);
    }
    shared_data[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = nt / 2u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        broadcast_val = shared_data[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = 1.0f / broadcast_val;
    for (uint i = tid; i < D; i += nt) {
        out_ptr[i] = exp(row_ptr[i] - row_max) * inv_sum;
    }
}
