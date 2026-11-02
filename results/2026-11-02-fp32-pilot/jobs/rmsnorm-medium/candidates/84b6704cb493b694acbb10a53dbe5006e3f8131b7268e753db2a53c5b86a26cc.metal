#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint tg [[threadgroup_position_in_grid]], uint tpsg [[threads_per_simdgroup]], uint tptg [[threads_per_threadgroup]]) {
    uint rows_per_group = tptg / tpsg;
    uint row = tg * rows_per_group + sg;
    if (row >= B) {
        return;
    }
    if (D == 1024u && tpsg == 32u) {
        device const float4* x4 = (device const float4*)(x + row * 1024u);
        device const float4* p4 = (device const float4*)parameter;
        device float4* o4 = (device float4*)(out + row * 1024u);
        float4 v0 = x4[lane];
        float4 v1 = x4[lane + 32u];
        float4 v2 = x4[lane + 64u];
        float4 v3 = x4[lane + 96u];
        float4 v4 = x4[lane + 128u];
        float4 v5 = x4[lane + 160u];
        float4 v6 = x4[lane + 192u];
        float4 v7 = x4[lane + 224u];
        float sum = dot(v0, v0) + dot(v1, v1) + dot(v2, v2) + dot(v3, v3) + dot(v4, v4) + dot(v5, v5) + dot(v6, v6) + dot(v7, v7);
        sum = simd_sum(sum);
        float inv = rsqrt(sum * (1.0f / 1024.0f) + epsilon);
        o4[lane] = v0 * inv * p4[lane];
        o4[lane + 32u] = v1 * inv * p4[lane + 32u];
        o4[lane + 64u] = v2 * inv * p4[lane + 64u];
        o4[lane + 96u] = v3 * inv * p4[lane + 96u];
        o4[lane + 128u] = v4 * inv * p4[lane + 128u];
        o4[lane + 160u] = v5 * inv * p4[lane + 160u];
        o4[lane + 192u] = v6 * inv * p4[lane + 192u];
        o4[lane + 224u] = v7 * inv * p4[lane + 224u];
        return;
    }
    float sum = 0.0f;
    uint base = row * D;
    for (uint j = lane; j < D; j += tpsg) {
        float v = x[base + j];
        sum += v * v;
    }
    sum = simd_sum(sum);
    float inv = rsqrt(sum / float(D) + epsilon);
    for (uint j = lane; j < D; j += tpsg) {
        out[base + j] = x[base + j] * inv * parameter[j];
    }
}
