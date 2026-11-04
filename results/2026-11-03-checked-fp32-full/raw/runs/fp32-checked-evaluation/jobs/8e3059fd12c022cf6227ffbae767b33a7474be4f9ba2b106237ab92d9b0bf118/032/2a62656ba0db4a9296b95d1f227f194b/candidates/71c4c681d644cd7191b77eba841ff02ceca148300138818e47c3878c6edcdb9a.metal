#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tpos [[thread_position_in_threadgroup]], uint3 gpos [[threadgroup_position_in_grid]], uint3 tsize [[threads_per_threadgroup]]) {
    (void)epsilon;
    uint tid = tpos.x;
    uint row = gpos.x;
    uint tg = tsize.x;
    if (row >= B) { return; }
    threadgroup float sdata[256];
    float local_max = -3.4028235e38f;
    for (uint i = tid; i < D; i += tg) {
        float v = x[row * D + i];
        local_max = max(local_max, v);
    }
    sdata[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) { sdata[tid] = max(sdata[tid], sdata[tid + stride]); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = sdata[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_sum = 0.0f;
    for (uint i = tid; i < D; i += tg) {
        float v = x[row * D + i];
        local_sum += exp(v - row_max);
    }
    sdata[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) { sdata[tid] += sdata[tid + stride]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_sum = sdata[0];
    float inv_sum = 1.0f / row_sum;
    for (uint i = tid; i < D; i += tg) {
        float v = x[row * D + i];
        out[row * D + i] = exp(v - row_max) * inv_sum;
    }
}
