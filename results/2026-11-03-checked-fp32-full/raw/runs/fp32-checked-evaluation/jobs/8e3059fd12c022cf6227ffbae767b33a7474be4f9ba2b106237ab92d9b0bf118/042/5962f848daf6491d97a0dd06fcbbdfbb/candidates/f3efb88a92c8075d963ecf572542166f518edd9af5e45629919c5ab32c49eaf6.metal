#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint tg_pos [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]], uint tg_size [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint simd_id [[simdgroup_index_in_threadgroup]], uint sg_width [[threads_per_simdgroup]]) {
    uint simdgroups = tg_size / sg_width;
    uint row = tg_pos * simdgroups + simd_id;
    if (row >= B) return;
    uint base = row * D;
    uint D4 = D >> 2;
    uint tail = D & 3u;
    uint per_lane = (D4 + sg_width - 1u) / sg_width;
    if (per_lane <= 32u) {
        float4 buf[32];
        device const float4* x4 = (device const float4*)(x + base);
        device const float4* p4 = (device const float4*)parameter;
        device float4* o4 = (device float4*)(out + base);
        float sum = 0.0f;
        for (uint k = 0u; k < 32u; ++k) {
            uint idx = lane + k * sg_width;
            float4 v = float4(0.0f);
            if (idx < D4) {
                v = x4[idx];
                sum += dot(v, v);
            }
            buf[k] = v;
        }
        float tail_val = 0.0f;
        if (tail != 0u && lane < tail) {
            tail_val = x[base + (D4 << 2) + lane];
            sum += tail_val * tail_val;
        }
        float total = simd_broadcast_first(simd_sum(sum));
        float scale = rsqrt(total / float(D) + epsilon);
        for (uint k = 0u; k < 32u; ++k) {
            uint idx = lane + k * sg_width;
            if (idx < D4) o4[idx] = buf[k] * scale * p4[idx];
        }
        if (tail != 0u && lane < tail) {
            uint col = (D4 << 2) + lane;
            out[base + col] = tail_val * scale * parameter[col];
        }
    } else {
        float sum = 0.0f;
        for (uint i = lane; i < D; i += sg_width) {
            float v = x[base + i];
            sum += v * v;
        }
        float total = simd_broadcast_first(simd_sum(sum));
        float scale = rsqrt(total / float(D) + epsilon);
        for (uint i = lane; i < D; i += sg_width) {
            out[base + i] = x[base + i] * scale * parameter[i];
        }
    }
}
