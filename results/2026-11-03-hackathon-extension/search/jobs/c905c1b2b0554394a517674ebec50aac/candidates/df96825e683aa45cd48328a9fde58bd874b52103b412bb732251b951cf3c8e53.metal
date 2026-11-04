#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 gid [[thread_position_in_grid]], uint3 tg [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]], uint3 tgs [[threads_per_threadgroup]], uint3 grid_dim [[threads_per_grid]], uint simd_lane [[thread_index_in_simdgroup]], uint simd_id [[simdgroup_index_in_threadgroup]], uint simd_size [[threads_per_simdgroup]]) {
    uint total_simds = grid_dim.x / simd_size;
    if (total_simds == 0u) { total_simds = 1u; }
    uint first_row = gid.x / simd_size;
    for (uint row = first_row; row < B; row += total_simds) {
        uint base = row * D;
        if (D == 1024u && simd_size == 32u) {
            device const float4* x4 = reinterpret_cast<device const float4*>(x + base);
            device const float4* p4 = reinterpret_cast<device const float4*>(parameter);
            device float4* o4 = reinterpret_cast<device float4*>(out + base);
            float4 v[8];
            float sum = 0.0f;
            for (uint i = 0u; i < 8u; ++i) {
                uint idx = simd_lane + i * 32u;
                v[i] = x4[idx];
                sum += dot(v[i], v[i]);
            }
            float inv = rsqrt(simd_sum(sum) / 1024.0f + epsilon);
            for (uint i = 0u; i < 8u; ++i) {
                uint idx = simd_lane + i * 32u;
                o4[idx] = v[i] * inv * p4[idx];
            }
        } else {
            float sum = 0.0f;
            for (uint c = simd_lane; c < D; c += simd_size) {
                float val = x[base + c];
                sum += val * val;
            }
            float inv = rsqrt(simd_sum(sum) / float(D) + epsilon);
            for (uint c = simd_lane; c < D; c += simd_size) {
                out[base + c] = x[base + c] * inv * parameter[c];
            }
        }
    }
}
