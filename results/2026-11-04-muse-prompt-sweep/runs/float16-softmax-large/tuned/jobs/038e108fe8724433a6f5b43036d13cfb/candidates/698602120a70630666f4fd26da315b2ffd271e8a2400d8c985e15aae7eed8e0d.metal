#include <metal_stdlib>
using namespace metal;
inline float tg_max(float v, threadgroup float* s, uint lane, uint sg, uint nsg) {
    float r = simd_max(v);
    if (lane == 0u) s[sg] = r;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float r2 = (lane < nsg) ? s[lane] : -1.0e30f;
        r2 = simd_max(r2);
        if (lane == 0u) s[0] = r2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float outv = s[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return outv;
}
inline float tg_sum(float v, threadgroup float* s, uint lane, uint sg, uint nsg) {
    float r = simd_sum(v);
    if (lane == 0u) s[sg] = r;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float r2 = (lane < nsg) ? s[lane] : 0.0f;
        r2 = simd_sum(r2);
        if (lane == 0u) s[0] = r2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float outv = s[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return outv;
}
kernel void metalmind_kernel(device const half* x, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tgid [[threadgroup_position_in_grid]], uint3 ltid [[thread_position_in_threadgroup]], uint3 tgdim [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint sgsize [[threads_per_simdgroup]]) {
    (void)epsilon;
    uint row = tgid.x;
    if (row >= B) return;
    uint tid = ltid.x;
    uint nsg = tgdim.x / sgsize;
    threadgroup float s[32];
    uint base = row * D;
    if (D == 4096u) {
        float vals[16];
        float lmax = -1.0e30f;
        for (uint v = 0u; v < 4u; ++v) {
            uint e = (v << 10u) + (tid << 2u);
            half4 h = (*(device const half4*)(x + base + e));
            vals[v * 4u + 0u] = float(h.x);
            vals[v * 4u + 1u] = float(h.y);
            vals[v * 4u + 2u] = float(h.z);
            vals[v * 4u + 3u] = float(h.w);
            lmax = max(lmax, max(max(float(h.x), float(h.y)), max(float(h.z), float(h.w))));
        }
        float gmax = tg_max(lmax, s, lane, sg, nsg);
        float lsum = 0.0f;
        for (uint k = 0u; k < 16u; ++k) {
            vals[k] = exp(vals[k] - gmax);
            lsum += vals[k];
        }
        float gsum = tg_sum(lsum, s, lane, sg, nsg);
        float inv = 1.0f / gsum;
        for (uint v = 0u; v < 4u; ++v) {
            uint e = (v << 10u) + (tid << 2u);
            half4 o = half4(half(vals[v * 4u + 0u] * inv), half(vals[v * 4u + 1u] * inv), half(vals[v * 4u + 2u] * inv), half(vals[v * 4u + 3u] * inv));
            (*(device half4*)(out + base + e)) = o;
        }
        return;
    }
    float lmax = -1.0e30f;
    for (uint i = tid; i < D; i += tgdim.x) lmax = max(lmax, float(x[base + i]));
    float gmax = tg_max(lmax, s, lane, sg, nsg);
    float lsum = 0.0f;
    for (uint i = tid; i < D; i += tgdim.x) lsum += exp(float(x[base + i]) - gmax);
    float gsum = tg_sum(lsum, s, lane, sg, nsg);
    float inv = 1.0f / gsum;
    for (uint i = tid; i < D; i += tgdim.x) out[base + i] = half(exp(float(x[base + i]) - gmax) * inv);
}
