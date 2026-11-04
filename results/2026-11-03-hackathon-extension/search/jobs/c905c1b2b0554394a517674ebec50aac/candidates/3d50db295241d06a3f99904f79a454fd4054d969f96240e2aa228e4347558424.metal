#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 gid [[thread_position_in_grid]], uint3 tg [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]], uint3 tgs [[threads_per_threadgroup]], uint3 grid_dim [[threads_per_grid]], uint simd_lane [[thread_index_in_simdgroup]], uint simd_id [[simdgroup_index_in_threadgroup]], uint simd_size [[threads_per_simdgroup]]) {
    threadgroup float partials[32];
    uint row = tg.x;
    if (row >= B) { return; }
    uint tid = lid.x;
    uint nthreads = tgs.x;
    if (nthreads == 0u) { nthreads = 1u; }
    uint num_simds = (nthreads + simd_size - 1u) / simd_size;
    if (num_simds > 32u) { num_simds = 32u; }
    uint base = row * D;
    if (D == 1024u && tid < 256u) {
        device const float4* x4 = reinterpret_cast<device const float4*>(x + base);
        device const float4* p4 = reinterpret_cast<device const float4*>(parameter);
        device float4* o4 = reinterpret_cast<device float4*>(out + base);
        float4 v = x4[tid];
        float sum = simd_sum(dot(v, v));
        if (simd_lane == 0u && simd_id < 32u) { partials[simd_id] = sum; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = 0.0f;
        for (uint i = 0u; i < num_simds; ++i) { total += partials[i]; }
        float inv = rsqrt(total * (1.0f / 1024.0f) + epsilon);
        o4[tid] = v * inv * p4[tid];
    } else {
        float sum = 0.0f;
        for (uint c = tid; c < D; c += nthreads) { float val = x[base + c]; sum += val * val; }
        sum = simd_sum(sum);
        if (simd_lane == 0u && simd_id < 32u) { partials[simd_id] = sum; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = 0.0f;
        for (uint i = 0u; i < num_simds; ++i) { total += partials[i]; }
        float inv = (D == 0u) ? 0.0f : rsqrt(total / float(D) + epsilon);
        for (uint c = tid; c < D; c += nthreads) { out[base + c] = x[base + c] * inv * parameter[c]; }
    }
}
