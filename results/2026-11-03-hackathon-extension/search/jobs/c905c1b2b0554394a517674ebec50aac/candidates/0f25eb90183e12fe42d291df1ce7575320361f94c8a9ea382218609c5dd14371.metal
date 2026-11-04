#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 gid [[thread_position_in_grid]], uint3 tg [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]], uint3 tgs [[threads_per_threadgroup]], uint3 grid_size [[threads_in_grid]], uint simd_lane [[thread_index_in_simdgroup]], uint simd_id [[simdgroup_index_in_threadgroup]], uint simd_size [[threads_per_simdgroup]]) {
    uint simdgroups_per_tg = tgs.x / simd_size;
    uint global_simdgroup = tg.x * simdgroups_per_tg + simd_id;
    uint total_simdgroups = grid_size.x / simd_size;
    if (total_simdgroups == 0u) {
        return;
    }
    for (uint row = global_simdgroup; row < B; row += total_simdgroups) {
        uint base = row * D;
        float sum = 0.0f;
        if ((D & 3u) == 0u) {
            uint vec_count = D >> 2u;
            device const float4* x4 = reinterpret_cast<device const float4*>(x + base);
            for (uint i = simd_lane; i < vec_count; i += simd_size) {
                float4 v = x4[i];
                sum += dot(v, v);
            }
            sum = simd_sum(sum);
            float inv = rsqrt(sum / float(D) + epsilon);
            device const float4* p4 = reinterpret_cast<device const float4*>(parameter);
            device float4* o4 = reinterpret_cast<device float4*>(out + base);
            for (uint i = simd_lane; i < vec_count; i += simd_size) {
                float4 v = x4[i];
                o4[i] = v * inv * p4[i];
            }
        } else {
            for (uint c = simd_lane; c < D; c += simd_size) {
                float v = x[base + c];
                sum += v * v;
            }
            sum = simd_sum(sum);
            float inv = rsqrt(sum / float(D) + epsilon);
            for (uint c = simd_lane; c < D; c += simd_size) {
                out[base + c] = x[base + c] * inv * parameter[c];
            }
        }
    }
}
