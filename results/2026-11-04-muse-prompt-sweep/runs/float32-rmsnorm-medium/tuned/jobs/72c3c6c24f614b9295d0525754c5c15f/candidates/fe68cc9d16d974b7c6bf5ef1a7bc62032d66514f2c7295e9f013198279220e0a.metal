#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint simd_id [[simdgroup_index_in_threadgroup]], uint threads_per_tg [[threads_per_threadgroup]], uint simd_width [[threads_per_simdgroup]], uint tid [[thread_position_in_threadgroup]], uint tg_count [[threadgroups_per_grid]]) {
    threadgroup float4 wcache[256];
    uint sw = max(1u, simd_width);
    uint rows_per_tg = max(1u, threads_per_tg / sw);
    if (D == 1024u && simd_width == 32u) {
        for (uint i = tid; i < 256u; i += threads_per_tg) {
            wcache[i] = ((device const float4*)parameter)[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint rows_per_block = rows_per_tg * 2u;
        uint block_stride = tg_count * rows_per_block;
        for (uint block = tgid * rows_per_block; block < B; block += block_stride) {
            for (uint r = 0u; r < 2u; ++r) {
                uint row = block + r * rows_per_tg + simd_id;
                if (row < B) {
                    uint base4 = row << 8;
                    uint i0 = lane;
                    uint i1 = i0 + 32u;
                    uint i2 = i1 + 32u;
                    uint i3 = i2 + 32u;
                    uint i4 = i3 + 32u;
                    uint i5 = i4 + 32u;
                    uint i6 = i5 + 32u;
                    uint i7 = i6 + 32u;
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
                    float inv = rsqrt(sum * 0.0009765625f + epsilon);
                    float s = inv;
                    ((device float4*)out)[base4 + i0] = v0 * s * wcache[i0];
                    ((device float4*)out)[base4 + i1] = v1 * s * wcache[i1];
                    ((device float4*)out)[base4 + i2] = v2 * s * wcache[i2];
                    ((device float4*)out)[base4 + i3] = v3 * s * wcache[i3];
                    ((device float4*)out)[base4 + i4] = v4 * s * wcache[i4];
                    ((device float4*)out)[base4 + i5] = v5 * s * wcache[i5];
                    ((device float4*)out)[base4 + i6] = v6 * s * wcache[i6];
                    ((device float4*)out)[base4 + i7] = v7 * s * wcache[i7];
                }
            }
        }
        return;
    }
    uint total_simd = tg_count * rows_per_tg;
    for (uint row = tgid * rows_per_tg + simd_id; row < B; row += total_simd) {
        if ((D & 3u) == 0u) {
            uint d4 = D >> 2;
            uint base4 = row * d4;
            float sum = 0.0f;
            for (uint i = lane; i < d4; i += sw) {
                float4 v = ((device const float4*)x)[base4 + i];
                sum += dot(v, v);
            }
            sum = simd_sum(sum);
            float inv = rsqrt(sum / float(D) + epsilon);
            for (uint i = lane; i < d4; i += sw) {
                float4 v = ((device const float4*)x)[base4 + i];
                ((device float4*)out)[base4 + i] = v * inv * ((device const float4*)parameter)[i];
            }
        } else {
            uint base = row * D;
            float sum = 0.0f;
            for (uint i = lane; i < D; i += sw) {
                float v = x[base + i];
                sum += v * v;
            }
            sum = simd_sum(sum);
            float inv = rsqrt(sum / float(D) + epsilon);
            for (uint i = lane; i < D; i += sw) {
                out[base + i] = x[base + i] * inv * parameter[i];
            }
        }
    }
}
