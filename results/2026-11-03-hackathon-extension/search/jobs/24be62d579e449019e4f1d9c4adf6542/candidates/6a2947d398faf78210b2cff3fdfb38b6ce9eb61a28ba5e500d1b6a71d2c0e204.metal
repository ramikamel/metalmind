#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint tgs [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sgid [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]]) {
    (void)epsilon;
    if (tg >= B || D == 0) return;
    uint base = tg * D;
    threadgroup float red[64];
    if (D == 4096u && tgs == 256u) {
        device const float4* x4 = (device const float4*)(x + base);
        device float4* o4 = (device float4*)(out + base);
        float4 a = x4[tid];
        float4 b = x4[tid + 256u];
        float4 c = x4[tid + 512u];
        float4 d = x4[tid + 768u];
        float local_max = max(max(a.x, a.y), max(a.z, a.w));
        local_max = max(local_max, max(max(b.x, b.y), max(b.z, b.w)));
        local_max = max(local_max, max(max(c.x, c.y), max(c.z, c.w)));
        local_max = max(local_max, max(max(d.x, d.y), max(d.z, d.w)));
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
        a = exp(a - row_max);
        b = exp(b - row_max);
        c = exp(c - row_max);
        d = exp(d - row_max);
        float local_sum = ((a.x + a.y) + (a.z + a.w)) + ((b.x + b.y) + (b.z + b.w)) + ((c.x + c.y) + (c.z + c.w)) + ((d.x + d.y) + (d.z + d.w));
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
        o4[tid] = a * inv_sum;
        o4[tid + 256u] = b * inv_sum;
        o4[tid + 512u] = c * inv_sum;
        o4[tid + 768u] = d * inv_sum;
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
