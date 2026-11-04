#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint tgs [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sgid [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]]) {
    (void)epsilon;
    if (tg >= B || D == 0) return;
    uint row = tg;
    uint base = row * D;
    threadgroup float red[64];
    if (D == 4096 && tgs == 512) {
        device const float4* x4 = (device const float4*)(x + base);
        device float4* o4 = (device float4*)(out + base);
        uint i0 = tid;
        uint i1 = tid + 512;
        float4 v0 = x4[i0];
        float4 v1 = x4[i1];
        float local_max = max(max(v0.x, v0.y), max(v0.z, v0.w));
        local_max = max(local_max, max(max(v1.x, v1.y), max(v1.z, v1.w)));
        float m = simd_max(local_max);
        if (lane == 0) red[sgid] = m;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float mm = red[0];
            for (uint i = 1; i < nsg; ++i) mm = max(mm, red[i]);
            red[0] = mm;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float row_max = red[0];
        float4 e0 = exp(v0 - row_max);
        float4 e1 = exp(v1 - row_max);
        float local_sum = ((e0.x + e0.y) + (e0.z + e0.w)) + ((e1.x + e1.y) + (e1.z + e1.w));
        float s = simd_sum(local_sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) red[sgid] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float ss = red[0];
            for (uint i = 1; i < nsg; ++i) ss += red[i];
            red[0] = ss;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv_sum = 1.0f / red[0];
        o4[i0] = e0 * inv_sum;
        o4[i1] = e1 * inv_sum;
        return;
    }
    float local_max = -3.402823466e+38f;
    for (uint i = tid; i < D; i += tgs) local_max = max(local_max, x[base + i]);
    float m = simd_max(local_max);
    if (lane == 0) red[sgid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float mm = red[0];
        for (uint i = 1; i < nsg; ++i) mm = max(mm, red[i]);
        red[0] = mm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = red[0];
    float local_sum = 0.0f;
    for (uint i = tid; i < D; i += tgs) local_sum += exp(x[base + i] - row_max);
    float s = simd_sum(local_sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) red[sgid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float ss = red[0];
        for (uint i = 1; i < nsg; ++i) ss += red[i];
        red[0] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = 1.0f / red[0];
    for (uint i = tid; i < D; i += tgs) out[base + i] = exp(x[base + i] - row_max) * inv_sum;
}
