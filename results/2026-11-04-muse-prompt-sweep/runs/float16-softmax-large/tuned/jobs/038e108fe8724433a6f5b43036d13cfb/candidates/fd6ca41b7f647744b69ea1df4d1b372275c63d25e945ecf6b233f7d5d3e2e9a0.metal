#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tgid [[threadgroup_position_in_grid]], uint3 ltid [[thread_position_in_threadgroup]], uint3 tgdim [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint sgsize [[threads_per_simdgroup]]) {
    (void)epsilon;
    uint row = tgid.x;
    if (row >= B) return;
    uint tid = ltid.x;
    uint nsg = tgdim.x / sgsize;
    threadgroup float smax[32];
    threadgroup float ssum[32];
    uint base = row * D;
    if (D == 4096u) {
        float vals[32];
        float lmax = -1.0e30f;
        for (uint v = 0u; v < 8u; ++v) {
            uint e = (v << 9u) + (tid << 2u);
            half4 h = (*(device const half4*)(x + base + e));
            vals[v * 4u + 0u] = float(h.x);
            vals[v * 4u + 1u] = float(h.y);
            vals[v * 4u + 2u] = float(h.z);
            vals[v * 4u + 3u] = float(h.w);
            lmax = max(lmax, max(max(float(h.x), float(h.y)), max(float(h.z), float(h.w))));
        }
        float r = simd_max(lmax);
        if (lane == 0u) smax[sg] = r;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float r2 = (lane < nsg) ? smax[lane] : -1.0e30f;
            r2 = simd_max(r2);
            if (lane == 0u) smax[0] = r2;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float gmax = smax[0];
        float lsum = 0.0f;
        for (uint k = 0u; k < 32u; ++k) {
            vals[k] = exp2((vals[k] - gmax) * 1.4426950408889634f);
            lsum += vals[k];
        }
        float rs = simd_sum(lsum);
        if (lane == 0u) ssum[sg] = rs;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float r2 = (lane < nsg) ? ssum[lane] : 0.0f;
            r2 = simd_sum(r2);
            if (lane == 0u) ssum[0] = r2;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv = 1.0f / ssum[0];
        for (uint v = 0u; v < 8u; ++v) {
            uint e = (v << 9u) + (tid << 2u);
            half4 o = half4(half(vals[v * 4u + 0u] * inv), half(vals[v * 4u + 1u] * inv), half(vals[v * 4u + 2u] * inv), half(vals[v * 4u + 3u] * inv));
            (*(device half4*)(out + base + e)) = o;
        }
        return;
    }
    float lmax = -1.0e30f;
    for (uint i = tid; i < D; i += tgdim.x) lmax = max(lmax, float(x[base + i]));
    float r = simd_max(lmax);
    if (lane == 0u) smax[sg] = r;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float r2 = (lane < nsg) ? smax[lane] : -1.0e30f;
        r2 = simd_max(r2);
        if (lane == 0u) smax[0] = r2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gmax = smax[0];
    float lsum = 0.0f;
    for (uint i = tid; i < D; i += tgdim.x) lsum += exp(float(x[base + i]) - gmax);
    float rs = simd_sum(lsum);
    if (lane == 0u) ssum[sg] = rs;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float r2 = (lane < nsg) ? ssum[lane] : 0.0f;
        r2 = simd_sum(r2);
        if (lane == 0u) ssum[0] = r2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = 1.0f / ssum[0];
    for (uint i = tid; i < D; i += tgdim.x) out[base + i] = half(exp(float(x[base + i]) - gmax) * inv);
}
