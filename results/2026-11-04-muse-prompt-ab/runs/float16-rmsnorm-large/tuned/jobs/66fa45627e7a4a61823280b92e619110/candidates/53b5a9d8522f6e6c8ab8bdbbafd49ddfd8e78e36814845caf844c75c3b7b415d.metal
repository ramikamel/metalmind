#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device const half* parameter, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint tptg [[threads_per_threadgroup]], uint sptg [[threads_per_simdgroup]]) {
    if (row >= B) return;
    device const half4* x4 = (device const half4*)x;
    device const half4* w4 = (device const half4*)parameter;
    device half4* o4 = (device half4*)out;
    uint rowBase = row * D;
    uint rowBase4 = row * (D >> 2);
    half4 vals[4];
    float sumsq = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        uint col = (tid + i * tptg) * 4;
        half4 v = half4(0.0h);
        if (col + 3 < D) {
            v = x4[rowBase4 + (col >> 2)];
        } else if (col < D) {
            if (col < D) v.x = x[rowBase + col];
            if (col + 1 < D) v.y = x[rowBase + col + 1];
            if (col + 2 < D) v.z = x[rowBase + col + 2];
            if (col + 3 < D) v.w = x[rowBase + col + 3];
        }
        vals[i] = v;
        float4 f = float4(v);
        sumsq += dot(f, f);
    }
    uint covered = tptg * 16;
    for (uint col = covered + tid; col < D; col += tptg) {
        float f = float(x[rowBase + col]);
        sumsq += f * f;
    }
    threadgroup float partial[32];
    float sgSum = simd_sum(sumsq);
    if (lane == 0) partial[sg] = sgSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint nsg = tptg / sptg;
    float total = 0.0f;
    for (uint i = 0; i < nsg; ++i) total += partial[i];
    float inv = rsqrt(total / float(D) + epsilon);
    for (uint i = 0; i < 4; ++i) {
        uint col = (tid + i * tptg) * 4;
        if (col + 3 < D) {
            float4 f = float4(vals[i]);
            float4 w = float4(w4[col >> 2]);
            o4[rowBase4 + (col >> 2)] = half4(f * inv * w);
        } else if (col < D) {
            float4 f = float4(vals[i]);
            float4 w = float4(0.0f);
            w.x = float(parameter[col]);
            if (col + 1 < D) w.y = float(parameter[col + 1]);
            if (col + 2 < D) w.z = float(parameter[col + 2]);
            if (col + 3 < D) w.w = float(parameter[col + 3]);
            float4 y = f * inv * w;
            out[rowBase + col] = half(y.x);
            if (col + 1 < D) out[rowBase + col + 1] = half(y.y);
            if (col + 2 < D) out[rowBase + col + 2] = half(y.z);
            if (col + 3 < D) out[rowBase + col + 3] = half(y.w);
        }
    }
    for (uint col = covered + tid; col < D; col += tptg) {
        out[rowBase + col] = half(float(x[rowBase + col]) * inv * float(parameter[col]));
    }
}
