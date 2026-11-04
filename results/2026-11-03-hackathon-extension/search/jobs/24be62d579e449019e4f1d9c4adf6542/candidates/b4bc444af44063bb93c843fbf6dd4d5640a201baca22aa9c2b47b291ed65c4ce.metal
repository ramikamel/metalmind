#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint tgs [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sgid [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]]) {
    (void)epsilon;
    if (tg >= B || D == 0) return;
    uint row = tg;
    uint base = row * D;
    threadgroup float red[64];
    if (D == 4096 && tgs == 256) {
        device const float4* x4 = (device const float4*)(x + base);
        device float4* o4 = (device float4*)(out + base);
        float4 vals[4];
        float local_max = -3.402823466e+38f;
        for (uint k = 0; k < 4; ++k) {
            float4 v = x4[tid + k * 256];
            vals[k] = v;
            local_max = max(local_max, max(max(v.x, v.y), max(v.z, v.w)));
        }
        float m = simd_max(local_max);
        if (lane == 0) red[sgid] = m;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0) {
            float v = (lane < nsg) ? red[lane] : -3.402823466e+38f;
            v = simd_max(v);
            if (lane == 0) red[0] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float row_max = red[0];
        float4 rm = float4(row_max);
        float local_sum = 0.0f;
        for (uint k = 0; k < 4; ++k) {
            float4 e = exp(vals[k] - rm);
            vals[k] = e;
            local_sum += (e.x + e.y) + (e.z + e.w);
        }
        float s = simd_sum(local_sum);
        if (lane == 0) red[sgid] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0) {
            float v = (lane < nsg) ? red[lane] : 0.0f;
            v = simd_sum(v);
            if (lane == 0) red[0] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv_sum = 1.0f / red[0];
        float4 iv = float4(inv_sum);
        for (uint k = 0; k < 4; ++k) {
            o4[tid + k * 256] = vals[k] * iv;
        }
        return;
    }
    float local_max = -3.402823466e+38f;
    for (uint i = tid; i < D; i += tgs) local_max = max(local_max, x[base + i]);
    float m = simd_max(local_max);
    if (lane == 0) red[sgid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        float v = (lane < nsg) ? red[lane] : -3.402823466e+38f;
        v = simd_max(v);
        if (lane == 0) red[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = red[0];
    float local_sum = 0.0f;
    for (uint i = tid; i < D; i += tgs) local_sum += exp(x[base + i] - row_max);
    float s = simd_sum(local_sum);
    if (lane == 0) red[sgid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        float v = (lane < nsg) ? red[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) red[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = 1.0f / red[0];
    for (uint i = tid; i < D; i += tgs) out[base + i] = float(exp(x[base + i] - row_max) * inv_sum);
}
