#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint gid [[thread_position_in_grid]], uint row [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]], uint tgsize [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint simdSize [[threads_per_simdgroup]]) {
    (void)gid;
    (void)epsilon;
    if (row >= B) return;
    if (D == 0u) return;
    threadgroup float red[32];
    uint base = row * D;
    uint numSg = (tgsize + simdSize - 1u) / simdSize;
    if (D == 1024u) {
        device const float4* x4 = (device const float4*)x;
        device float4* o4 = (device float4*)out;
        uint vbase = row * 256u;
        float4 v0 = x4[vbase + lid];
        float4 v1 = x4[vbase + lid + 128u];
        float local_max = max(max(v0.x, v0.y), max(v0.z, v0.w));
        local_max = max(local_max, max(max(v1.x, v1.y), max(v1.z, v1.w)));
        float sg_max = simd_max(local_max);
        if (lane == 0u) red[sg] = sg_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float v = (lane < numSg) ? red[lane] : -3.4028234663852886e+38f;
            v = simd_max(v);
            if (lane == 0u) red[0] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float row_max = red[0];
        float4 e0 = exp(v0 - row_max);
        float4 e1 = exp(v1 - row_max);
        float local_sum = (e0.x + e0.y + e0.z + e0.w) + (e1.x + e1.y + e1.z + e1.w);
        float sg_sum = simd_sum(local_sum);
        if (lane == 0u) red[sg] = sg_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float v = (lane < numSg) ? red[lane] : 0.0f;
            v = simd_sum(v);
            if (lane == 0u) red[0] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv_sum = 1.0f / red[0];
        o4[vbase + lid] = e0 * inv_sum;
        o4[vbase + lid + 128u] = e1 * inv_sum;
        return;
    }
    float local_max = -3.4028234663852886e+38f;
    if ((D & 3u) == 0u) {
        device const float4* x4 = (device const float4*)x;
        uint vecD = D >> 2u;
        uint vbase = row * vecD;
        for (uint i = lid; i < vecD; i += tgsize) {
            float4 v = x4[vbase + i];
            local_max = max(local_max, max(max(v.x, v.y), max(v.z, v.w)));
        }
    } else {
        for (uint i = lid; i < D; i += tgsize) local_max = max(local_max, x[base + i]);
    }
    float sg_max = simd_max(local_max);
    if (lane == 0u) red[sg] = sg_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float v = (lane < numSg) ? red[lane] : -3.4028234663852886e+38f;
        v = simd_max(v);
        if (lane == 0u) red[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = red[0];
    float local_sum = 0.0f;
    for (uint i = lid; i < D; i += tgsize) {
        float v = exp(x[base + i] - row_max);
        out[base + i] = v;
        local_sum += v;
    }
    float sg_sum = simd_sum(local_sum);
    if (lane == 0u) red[sg] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float v = (lane < numSg) ? red[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0u) red[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = 1.0f / red[0];
    for (uint i = lid; i < D; i += tgsize) out[base + i] = out[base + i] * inv_sum;
}
