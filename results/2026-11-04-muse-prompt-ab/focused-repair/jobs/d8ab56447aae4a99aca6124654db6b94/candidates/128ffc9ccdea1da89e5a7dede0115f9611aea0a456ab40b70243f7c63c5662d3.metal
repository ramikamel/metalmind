#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint W [[threads_per_simdgroup]], uint T [[threads_per_threadgroup]]) {
    (void)epsilon;
    (void)lane;
    (void)W;
    if (row >= B) return;
    uint base = row * D;
    if (D == 1024u && T == 32u) {
        device const half4* xv = (device const half4*)(x + base);
        half4 vals[8];
        float lm = -3.4028235e38f;
        for (uint k = 0u; k < 8u; ++k) {
            half4 h = xv[tid + 32u * k];
            vals[k] = h;
            lm = max(lm, max(max(float(h.x), float(h.y)), max(float(h.z), float(h.w))));
        }
        float rmax = simd_max(lm);
        float ev[32];
        float ls = 0.0f;
        for (uint k = 0u; k < 8u; ++k) {
            half4 h = vals[k];
            float e0 = exp(float(h.x) - rmax);
            float e1 = exp(float(h.y) - rmax);
            float e2 = exp(float(h.z) - rmax);
            float e3 = exp(float(h.w) - rmax);
            uint o = 4u * k;
            ev[o] = e0;
            ev[o + 1u] = e1;
            ev[o + 2u] = e2;
            ev[o + 3u] = e3;
            ls += ((e0 + e1) + (e2 + e3));
        }
        float rs = simd_sum(ls);
        float inv = (rs > 0.0f) ? (1.0f / rs) : 0.0f;
        device half4* ov = (device half4*)(out + base);
        for (uint k = 0u; k < 8u; ++k) {
            uint o = 4u * k;
            ov[tid + 32u * k] = half4(half(ev[o] * inv), half(ev[o + 1u] * inv), half(ev[o + 2u] * inv), half(ev[o + 3u] * inv));
        }
        return;
    }
    float lm = -3.4028235e38f;
    for (uint j = tid; j < D; j += T) lm = max(lm, float(x[base + j]));
    float rmax = simd_max(lm);
    float ls = 0.0f;
    for (uint j = tid; j < D; j += T) ls += exp(float(x[base + j]) - rmax);
    float rs = simd_sum(ls);
    float inv = (rs > 0.0f) ? (1.0f / rs) : 0.0f;
    for (uint j = tid; j < D; j += T) out[base + j] = half(exp(float(x[base + j]) - rmax) * inv);
}
