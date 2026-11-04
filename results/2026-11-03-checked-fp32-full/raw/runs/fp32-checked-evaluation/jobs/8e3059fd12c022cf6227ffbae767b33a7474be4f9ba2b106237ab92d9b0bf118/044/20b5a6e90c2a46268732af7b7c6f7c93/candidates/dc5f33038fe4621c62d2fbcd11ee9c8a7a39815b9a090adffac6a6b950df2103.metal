#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint tid [[thread_index_in_threadgroup]], uint row [[threadgroup_position_in_grid]], uint gtid [[thread_position_in_grid]]) {
    (void)epsilon;
    (void)gtid;
    if (row >= B) return;
    threadgroup float red[256];
    uint base = row * D;
    float local_max = -3.402823466e+38f;
    for (uint i = tid; i < D; i += 256u) {
        local_max = max(local_max, x[base + i]);
    }
    red[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) red[tid] = max(red[tid], red[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_sum = 0.0f;
    for (uint i = tid; i < D; i += 256u) {
        local_sum += exp(x[base + i] - row_max);
    }
    red[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) red[tid] += red[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < D; i += 256u) {
        out[base + i] = exp(x[base + i] - row_max) * inv_sum;
    }
}
