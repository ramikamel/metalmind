#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint simd_id [[simdgroup_index_in_threadgroup]], uint threads_per_tg [[threads_per_threadgroup]], uint simd_width [[threads_per_simdgroup]]) {
    uint rows_per_tg = threads_per_tg / simd_width;
    uint row = tgid * rows_per_tg + simd_id;
    if (row >= B) {
        return;
    }
    uint D4 = D >> 2;
    if ((D & 3u) == 0u && D4 == (simd_width << 3)) {
        uint base4 = row * D4;
        uint i0 = lane;
        uint i1 = i0 + simd_width;
        uint i2 = i1 + simd_width;
        uint i3 = i2 + simd_width;
        uint i4 = i3 + simd_width;
        uint i5 = i4 + simd_width;
        uint i6 = i5 + simd_width;
        uint i7 = i6 + simd_width;
        float4 v0 = ((device const float4*)x)[base4 + i0];
        float4 v1 = ((device const float4*)x)[base4 + i1];
        float4 v2 = ((device const float4*)x)[base4 + i2];
        float4 v3 = ((device const float4*)x)[base4 + i3];
        float4 v4 = ((device const float4*)x)[base4 + i4];
        float4 v5 = ((device const float4*)x)[base4 + i5];
        float4 v6 = ((device const float4*)x)[base4 + i6];
        float4 v7 = ((device const float4*)x)[base4 + i7];
        float sum = dot(v0, v0) + dot(v1, v1) + dot(v2, v2) + dot(v3, v3) + dot(v4, v4) + dot(v5, v5) + dot(v6, v6) + dot(v7, v7);
        sum = simd_sum(sum);
        float inv = rsqrt(sum / float(D) + epsilon);
        ((device float4*)out)[base4 + i0] = v0 * inv * ((device const float4*)parameter)[i0];
        ((device float4*)out)[base4 + i1] = v1 * inv * ((device const float4*)parameter)[i1];
        ((device float4*)out)[base4 + i2] = v2 * inv * ((device const float4*)parameter)[i2];
        ((device float4*)out)[base4 + i3] = v3 * inv * ((device const float4*)parameter)[i3];
        ((device float4*)out)[base4 + i4] = v4 * inv * ((device const float4*)parameter)[i4];
        ((device float4*)out)[base4 + i5] = v5 * inv * ((device const float4*)parameter)[i5];
        ((device float4*)out)[base4 + i6] = v6 * inv * ((device const float4*)parameter)[i6];
        ((device float4*)out)[base4 + i7] = v7 * inv * ((device const float4*)parameter)[i7];
        return;
    }
    if ((D & 3u) == 0u) {
        uint base4 = row * D4;
        float sum = 0.0f;
        for (uint i = lane; i < D4; i += simd_width) {
            float4 v = ((device const float4*)x)[base4 + i];
            sum += dot(v, v);
        }
        sum = simd_sum(sum);
        float inv = rsqrt(sum / float(D) + epsilon);
        for (uint i = lane; i < D4; i += simd_width) {
            float4 v = ((device const float4*)x)[base4 + i];
            ((device float4*)out)[base4 + i] = v * inv * ((device const float4*)parameter)[i];
        }
        return;
    }
    uint base = row * D;
    float sum = 0.0f;
    for (uint i = lane; i < D; i += simd_width) {
        float v = x[base + i];
        sum += v * v;
    }
    sum = simd_sum(sum);
    float inv = rsqrt(sum / float(D) + epsilon);
    for (uint i = lane; i < D; i += simd_width) {
        out[base + i] = x[base + i] * inv * parameter[i];
    }
}
