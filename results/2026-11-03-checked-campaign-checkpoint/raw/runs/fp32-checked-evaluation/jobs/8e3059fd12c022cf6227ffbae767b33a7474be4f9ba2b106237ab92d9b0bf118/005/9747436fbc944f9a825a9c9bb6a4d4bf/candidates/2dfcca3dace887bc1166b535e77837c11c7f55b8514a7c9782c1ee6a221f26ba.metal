#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 gid [[thread_position_in_grid]], uint3 gpid [[threadgroup_position_in_grid]], uint3 tid_in_group [[thread_position_in_threadgroup]], uint3 threads_per_group [[threads_per_threadgroup]]) {
    (void)gid;
    (void)epsilon;
    uint row = gpid.x;
    if (row >= B || D == 0u) {
        return;
    }
    uint tid = tid_in_group.x;
    uint tg = threads_per_group.x;
    threadgroup float red[256];
    float local_max = -3.402823466e+38f;
    for (uint i = tid; i < D; i += tg) {
        float v = x[row * D + i];
        local_max = max(local_max, v);
    }
    red[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            red[tid] = max(red[tid], red[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_sum = 0.0f;
    for (uint i = tid; i < D; i += tg) {
        local_sum += exp(x[row * D + i] - row_max);
    }
    red[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            red[tid] = red[tid] + red[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_sum = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < D; i += tg) {
        float e = exp(x[row * D + i] - row_max);
        out[row * D + i] = float(e / row_sum);
    }
}
