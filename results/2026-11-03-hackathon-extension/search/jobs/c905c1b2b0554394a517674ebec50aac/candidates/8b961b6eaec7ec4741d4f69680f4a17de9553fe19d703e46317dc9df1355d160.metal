#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 gid [[thread_position_in_grid]], uint3 tg [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]], uint3 tgs [[threads_per_threadgroup]], uint3 grid_dim [[threads_per_grid]], uint simd_lane [[thread_index_in_simdgroup]], uint simd_id [[simdgroup_index_in_threadgroup]], uint simd_size [[threads_per_simdgroup]]) {
    threadgroup float tg_sums[64];
    threadgroup float tg_inv;
    uint row = tg.x;
    if (row >= B) { return; }
    uint base = row * D;
    if (D == 1024u && tgs.x == 256u) {
        device const float4* x4 = reinterpret_cast<device const float4*>(x + base);
        device const float4* p4 = reinterpret_cast<device const float4*>(parameter);
        device float4* o4 = reinterpret_cast<device float4*>(out + base);
        float4 v = x4[lid.x];
        float total = simd_sum(dot(v, v));
        if (simd_lane == 0u && simd_id < 64u) { tg_sums[simd_id] = total; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_id == 0u) {
            uint nsimd = tgs.x / simd_size;
            float s = (simd_lane < nsimd) ? tg_sums[simd_lane] : 0.0f;
            float grand = simd_sum(s);
            if (simd_lane == 0u) { tg_inv = rsqrt(grand * 0.0009765625f + epsilon); }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv = tg_inv;
        o4[lid.x] = v * inv * p4[lid.x];
    } else {
        float sum = 0.0f;
        for (uint c = lid.x; c < D; c += tgs.x) { float val = x[base + c]; sum += val * val; }
        float total = simd_sum(sum);
        if (simd_lane == 0u && simd_id < 64u) { tg_sums[simd_id] = total; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_id == 0u) {
            uint nsimd = tgs.x / simd_size;
            if (nsimd > 64u) { nsimd = 64u; }
            float s = (simd_lane < nsimd) ? tg_sums[simd_lane] : 0.0f;
            float grand = simd_sum(s);
            if (simd_lane == 0u) { tg_inv = rsqrt(grand / float(D) + epsilon); }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv = tg_inv;
        for (uint c = lid.x; c < D; c += tgs.x) { out[base + c] = x[base + c] * inv * parameter[c]; }
    }
}