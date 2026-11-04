#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tg_pos [[threadgroup_position_in_grid]], uint3 threads_per_tg [[threads_per_threadgroup]], uint simd_lane [[thread_index_in_simdgroup]], uint simd_id [[simdgroup_index_in_threadgroup]], uint threads_per_simd [[threads_per_simdgroup]]) {
    uint simds_per_tg = threads_per_tg.x / threads_per_simd;
    uint row = tg_pos.x * simds_per_tg + simd_id;
    if (row >= B) {
        return;
    }
    uint base = row * D;
    float sum = 0.0f;
    for (uint col = simd_lane; col < D; col += threads_per_simd) {
        float v = x[base + col];
        sum += v * v;
    }
    sum = simd_sum(sum);
    float inv = rsqrt(sum / float(D) + epsilon);
    for (uint col = simd_lane; col < D; col += threads_per_simd) {
        float v = x[base + col];
        out[base + col] = float(v * inv * parameter[col]);
    }
}
