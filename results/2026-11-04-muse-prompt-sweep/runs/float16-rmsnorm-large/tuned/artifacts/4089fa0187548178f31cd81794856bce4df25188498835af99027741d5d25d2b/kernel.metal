#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device const half* parameter, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint tg_pos [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sgi [[simdgroup_index_in_threadgroup]], uint tpt [[threads_per_threadgroup]], uint tps [[threads_per_simdgroup]]) {
    if (tg_pos >= B) {
        return;
    }
    uint row = tg_pos;
    uint vecD = D >> 2;
    bool vecPath = ((D & 3u) == 0u) && (vecD <= (tpt << 2));
    half4 vals[4];
    for (uint i = 0; i < 4; ++i) {
        vals[i] = half4(0);
    }
    float partial = 0.0f;
    if (vecPath) {
        uint baseVec = row * vecD;
        for (uint i = 0; i < 4; ++i) {
            uint idx = tid + i * tpt;
            if (idx < vecD) {
                half4 v = ((device const half4*)(x))[baseVec + idx];
                vals[i] = v;
                float4 f = float4(v);
                partial += dot(f, f);
            }
        }
    } else {
        uint base = row * D;
        for (uint c = tid; c < D; c += tpt) {
            float f = float(x[base + c]);
            partial += f * f;
        }
    }
    threadgroup float scratch[64];
    float sgSum = simd_sum(partial);
    if (lane == 0) {
        scratch[sgi] = sgSum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint numSg = (tpt + tps - 1) / tps;
    if (sgi == 0) {
        float v = (lane < numSg) ? scratch[lane] : 0.0f;
        float tot = simd_sum(v);
        if (lane == 0) {
            scratch[0] = tot;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = rsqrt(scratch[0] / float(D) + epsilon);
    if (vecPath) {
        uint baseVec = row * vecD;
        for (uint i = 0; i < 4; ++i) {
            uint idx = tid + i * tpt;
            if (idx < vecD) {
                float4 f = float4(vals[i]);
                float4 w = float4(((device const half4*)(parameter))[idx]);
                ((device half4*)(out))[baseVec + idx] = half4(f * inv * w);
            }
        }
    } else {
        uint base = row * D;
        for (uint c = tid; c < D; c += tpt) {
            float f = float(x[base + c]);
            float w = float(parameter[c]);
            out[base + c] = half(f * inv * w);
        }
    }
}
