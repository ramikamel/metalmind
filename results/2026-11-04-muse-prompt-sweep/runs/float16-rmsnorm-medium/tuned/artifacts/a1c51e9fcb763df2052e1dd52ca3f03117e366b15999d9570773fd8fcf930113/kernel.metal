#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device const half* parameter, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tg_pos [[threadgroup_position_in_grid]], uint3 tp_tg [[thread_position_in_threadgroup]], uint3 tptg [[threads_per_threadgroup]], uint lid [[thread_index_in_simdgroup]], uint sgid [[simdgroup_index_in_threadgroup]], uint sptg [[simdgroups_per_threadgroup]], uint spsg [[threads_per_simdgroup]]) {
    uint row = tg_pos.x * sptg + sgid;
    if (row >= B) {
        return;
    }
    if (D == 1024u && spsg == 32u) {
        uint base = row * 256u + lid;
        half4 c0 = ((device const half4*)x)[base];
        half4 c1 = ((device const half4*)x)[base + 32u];
        half4 c2 = ((device const half4*)x)[base + 64u];
        half4 c3 = ((device const half4*)x)[base + 96u];
        half4 c4 = ((device const half4*)x)[base + 128u];
        half4 c5 = ((device const half4*)x)[base + 160u];
        half4 c6 = ((device const half4*)x)[base + 192u];
        half4 c7 = ((device const half4*)x)[base + 224u];
        float4 v0 = float4(c0);
        float4 v1 = float4(c1);
        float4 v2 = float4(c2);
        float4 v3 = float4(c3);
        float4 v4 = float4(c4);
        float4 v5 = float4(c5);
        float4 v6 = float4(c6);
        float4 v7 = float4(c7);
        float sum = dot(v0, v0) + dot(v1, v1) + dot(v2, v2) + dot(v3, v3) + dot(v4, v4) + dot(v5, v5) + dot(v6, v6) + dot(v7, v7);
        float scale = rsqrt(simd_sum(sum) / 1024.0f + epsilon);
        ((device half4*)out)[base] = half4(v0 * scale * float4(((device const half4*)parameter)[lid]));
        ((device half4*)out)[base + 32u] = half4(v1 * scale * float4(((device const half4*)parameter)[lid + 32u]));
        ((device half4*)out)[base + 64u] = half4(v2 * scale * float4(((device const half4*)parameter)[lid + 64u]));
        ((device half4*)out)[base + 96u] = half4(v3 * scale * float4(((device const half4*)parameter)[lid + 96u]));
        ((device half4*)out)[base + 128u] = half4(v4 * scale * float4(((device const half4*)parameter)[lid + 128u]));
        ((device half4*)out)[base + 160u] = half4(v5 * scale * float4(((device const half4*)parameter)[lid + 160u]));
        ((device half4*)out)[base + 192u] = half4(v6 * scale * float4(((device const half4*)parameter)[lid + 192u]));
        ((device half4*)out)[base + 224u] = half4(v7 * scale * float4(((device const half4*)parameter)[lid + 224u]));
        return;
    }
    float sum = 0.0f;
    for (uint i = lid; i < D; i += spsg) {
        float v = float(x[row * D + i]);
        sum += v * v;
    }
    float scale = rsqrt(simd_sum(sum) / float(D) + epsilon);
    for (uint i = lid; i < D; i += spsg) {
        out[row * D + i] = half(float(x[row * D + i]) * scale * float(parameter[i]));
    }
}
