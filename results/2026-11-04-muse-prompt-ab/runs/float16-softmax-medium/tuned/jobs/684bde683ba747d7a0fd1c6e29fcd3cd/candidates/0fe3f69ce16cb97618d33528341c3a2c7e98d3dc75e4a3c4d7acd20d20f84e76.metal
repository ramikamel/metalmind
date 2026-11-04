#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint W [[threads_per_simdgroup]], uint T [[threads_per_threadgroup]]) {
    (void)epsilon;
    if (row >= B) return;
    uint base = row * D;
    threadgroup float sbuf[32];
    uint nsg = (W == 0u) ? 1u : (T + W - 1u) / W;
    if (D == 1024u && T == 256u) {
        device const half4* xv = (device const half4*)(x + base);
        half4 h = xv[tid];
        float a = float(h.x);
        float b = float(h.y);
        float c = float(h.z);
        float d = float(h.w);
        float lm = max(max(a, b), max(c, d));
        float sm = simd_max(lm);
        if (lane == 0u) sbuf[sg] = sm;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float v = (lane < nsg) ? sbuf[lane] : -3.4028235e38f;
            float rm = simd_max(v);
            if (lane == 0u) sbuf[0] = rm;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float rmax = sbuf[0];
        float ea = exp(a - rmax);
        float eb = exp(b - rmax);
        float ec = exp(c - rmax);
        float ed = exp(d - rmax);
        float ls = ((ea + eb) + (ec + ed));
        float ss = simd_sum(ls);
        if (lane == 0u) sbuf[sg] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float v = (lane < nsg) ? sbuf[lane] : 0.0f;
            float rs = simd_sum(v);
            if (lane == 0u) sbuf[0] = (rs > 0.0f) ? (1.0f / rs) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv = sbuf[0];
        device half4* ov = (device half4*)(out + base);
        ov[tid] = half4(half(ea * inv), half(eb * inv), half(ec * inv), half(ed * inv));
        return;
    }
    float lm = -3.4028235e38f;
    for (uint j = tid; j < D; j += T) lm = max(lm, float(x[base + j]));
    float sm = simd_max(lm);
    if (lane == 0u) sbuf[sg] = sm;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float v = (lane < nsg) ? sbuf[lane] : -3.4028235e38f;
        float rm = simd_max(v);
        if (lane == 0u) sbuf[0] = rm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rmax = sbuf[0];
    float ls = 0.0f;
    for (uint j = tid; j < D; j += T) ls += exp(float(x[base + j]) - rmax);
    float ss = simd_sum(ls);
    if (lane == 0u) sbuf[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float v = (lane < nsg) ? sbuf[lane] : 0.0f;
        float rs = simd_sum(v);
        if (lane == 0u) sbuf[0] = (rs > 0.0f) ? (1.0f / rs) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = sbuf[0];
    for (uint j = tid; j < D; j += T) out[base + j] = half(exp(float(x[base + j]) - rmax) * inv);
}
