#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tg_pos [[threadgroup_position_in_grid]], uint3 threads_per_tg [[threads_per_threadgroup]], uint simd_lane [[thread_index_in_simdgroup]], uint simd_id [[simdgroup_index_in_threadgroup]], uint threads_per_simd [[threads_per_simdgroup]]) {
    uint simds_per_tg = threads_per_tg.x / threads_per_simd;
    uint row = tg_pos.x * simds_per_tg + simd_id;
    if (row >= B) {
        return;
    }
    uint base = row * D;
    uint lane = simd_lane;
    uint W = threads_per_simd;
    if (D == 257u && W == 32u) {
        float v0 = x[base + lane];
        float v1 = x[base + lane + 32u];
        float v2 = x[base + lane + 64u];
        float v3 = x[base + lane + 96u];
        float v4 = x[base + lane + 128u];
        float v5 = x[base + lane + 160u];
        float v6 = x[base + lane + 192u];
        float v7 = x[base + lane + 224u];
        float p0 = parameter[lane];
        float p1 = parameter[lane + 32u];
        float p2 = parameter[lane + 64u];
        float p3 = parameter[lane + 96u];
        float p4 = parameter[lane + 128u];
        float p5 = parameter[lane + 160u];
        float p6 = parameter[lane + 192u];
        float p7 = parameter[lane + 224u];
        float vx = (lane == 0u) ? x[base + 256u] : 0.0f;
        float px = (lane == 0u) ? parameter[256u] : 0.0f;
        float sum = v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3 + v4 * v4 + v5 * v5 + v6 * v6 + v7 * v7 + vx * vx;
        sum = simd_sum(sum);
        float inv = rsqrt(sum * (1.0f / 257.0f) + epsilon);
        out[base + lane] = v0 * inv * p0;
        out[base + lane + 32u] = v1 * inv * p1;
        out[base + lane + 64u] = v2 * inv * p2;
        out[base + lane + 96u] = v3 * inv * p3;
        out[base + lane + 128u] = v4 * inv * p4;
        out[base + lane + 160u] = v5 * inv * p5;
        out[base + lane + 192u] = v6 * inv * p6;
        out[base + lane + 224u] = v7 * inv * p7;
        if (lane == 0u) {
            out[base + 256u] = vx * inv * px;
        }
        return;
    }
    float sum = 0.0f;
    for (uint col = lane; col < D; col += W) {
        float v = x[base + col];
        sum += v * v;
    }
    sum = simd_sum(sum);
    float inv = rsqrt(sum / float(D) + epsilon);
    for (uint col = lane; col < D; col += W) {
        float v = x[base + col];
        out[base + col] = v * inv * parameter[col];
    }
}
