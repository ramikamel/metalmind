#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint tid [[thread_position_in_threadgroup]], uint tg [[threadgroup_position_in_grid]], uint tgs [[threads_per_threadgroup]]) {
    (void)epsilon;
    uint row = tg;
    if (row >= B || D == 0) return;
    threadgroup float sdata[1024];
    device const float* xr = x + uint(row * D);
    device float* outr = out + uint(row * D);
    float local_max = -INFINITY;
    for (uint i = tid; i < D; i += tgs) {
        float v = xr[i];
        local_max = max(local_max, v);
    }
    if (tid < 1024) sdata[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = tgs >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) sdata[tid] = max(sdata[tid], sdata[tid + offset]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = sdata[0];
    float local_sum = 0.0f;
    for (uint i = tid; i < D; i += tgs) {
        local_sum += exp(xr[i] - row_max);
    }
    if (tid < 1024) sdata[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = tgs >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) sdata[tid] += sdata[tid + offset];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / sdata[0];
    for (uint i = tid; i < D; i += tgs) {
        float v = exp(xr[i] - row_max) * inv_sum;
        outr[i] = float(v);
    }
}
