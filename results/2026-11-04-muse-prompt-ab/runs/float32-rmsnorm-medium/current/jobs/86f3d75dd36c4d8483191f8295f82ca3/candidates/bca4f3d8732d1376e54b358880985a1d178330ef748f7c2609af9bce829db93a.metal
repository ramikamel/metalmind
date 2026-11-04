#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tg [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]], uint3 tgsize [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint sgw [[threads_per_simdgroup]]) {
    uint sg_per_tg = (sgw == 0u) ? 1u : (tgsize.x / sgw);
    if (sg_per_tg == 0u) { sg_per_tg = 1u; }
    uint row = tg.x * sg_per_tg + sg;
    if (row >= B) { return; }
    if (sgw == 0u) { sgw = 1u; }
    if ((D & 3u) == 0u) {
        uint d4 = D >> 2u;
        device const float4* x4 = reinterpret_cast<device const float4*>(x);
        device const float4* p4 = reinterpret_cast<device const float4*>(parameter);
        device float4* o4 = reinterpret_cast<device float4*>(out);
        uint base4 = row * d4;
        if (d4 <= (sgw << 3u)) {
            float4 vals[8];
            float s = 0.0f;
            uint k = 0u;
            for (uint i = lane; i < d4; i += sgw) {
                float4 v = x4[base4 + i];
                vals[k] = v;
                s += dot(v, v);
                k += 1u;
            }
            s = simd_sum(s);
            float inv = rsqrt(s / float(D) + epsilon);
            k = 0u;
            for (uint i = lane; i < d4; i += sgw) {
                o4[base4 + i] = vals[k] * inv * p4[i];
                k += 1u;
            }
            return;
        }
        float s = 0.0f;
        for (uint i = lane; i < d4; i += sgw) {
            float4 v = x4[base4 + i];
            s += dot(v, v);
        }
        s = simd_sum(s);
        float inv = rsqrt(s / float(D) + epsilon);
        for (uint i = lane; i < d4; i += sgw) {
            o4[base4 + i] = x4[base4 + i] * inv * p4[i];
        }
        return;
    }
    uint base = row * D;
    float s = 0.0f;
    for (uint i = lane; i < D; i += sgw) {
        float v = x[base + i];
        s += v * v;
    }
    s = simd_sum(s);
    float inv = rsqrt(s / float(D) + epsilon);
    for (uint i = lane; i < D; i += sgw) {
        out[base + i] = x[base + i] * inv * parameter[i];
    }
}
