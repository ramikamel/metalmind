#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tg_pos [[threadgroup_position_in_grid]], uint3 lid3 [[thread_position_in_threadgroup]], uint3 tg_size3 [[threads_per_threadgroup]]) {
    uint row = tg_pos.x;
    uint lid = lid3.x;
    uint tg = tg_size3.x;
    if (row >= B) {
        return;
    }
    device const float* row_x = x + (uint)(row * D);
    device float* row_out = out + (uint)(row * D);
    threadgroup float shared[256];
    float local_max = -3.402823466e+38f;
    for (uint i = lid; i < D; i += tg) {
        local_max = max(local_max, row_x[i]);
    }
    shared[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared[lid] = max(shared[lid], shared[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_sum = 0.0f;
    for (uint i = lid; i < D; i += tg) {
        local_sum += exp(row_x[i] - row_max);
    }
    shared[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared[lid] += shared[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / shared[0];
    for (uint i = lid; i < D; i += tg) {
        row_out[i] = float(exp(row_x[i] - row_max) * inv_sum);
    }
}
