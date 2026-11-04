#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device const half* parameter, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint tg_pos [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sgi [[simdgroup_index_in_threadgroup]], uint tpt [[threads_per_threadgroup]], uint tps [[threads_per_simdgroup]]) {
    if (tg_pos >= B) {
        return;
    }
    uint row = tg_pos;
    uint vec8D = D >> 3;
    bool fastPath = ((D & 7u) == 0u) && (vec8D <= (tpt << 1u)) && (vec8D > 0u);
    uint4 vals[2];
    vals[0] = uint4(0);
    vals[1] = uint4(0);
    float partial = 0.0f;
    if (fastPath) {
        uint baseVec = row * vec8D;
        for (uint i = 0; i < 2; ++i) {
            uint idx = tid + i * tpt;
            if (idx < vec8D) {
                uint4 v = ((device const uint4*)(x))[baseVec + idx];
                vals[i] = v;
                float4 flo = float4(as_half4(v.xy));
                float4 fhi = float4(as_half4(v.zw));
                partial += dot(flo, flo) + dot(fhi, fhi);
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
    if (fastPath) {
        uint baseVec = row * vec8D;
        for (uint i = 0; i < 2; ++i) {
            uint idx = tid + i * tpt;
            if (idx < vec8D) {
                uint4 v = vals[i];
                uint4 wv = ((device const uint4*)(parameter))[idx];
                float4 flo = float4(as_half4(v.xy));
                float4 fhi = float4(as_half4(v.zw));
                float4 wlo = float4(as_half4(wv.xy));
                float4 whi = float4(as_half4(wv.zw));
                half4 olo = half4(flo * inv * wlo);
                half4 ohi = half4(fhi * inv * whi);
                ((device uint4*)(out))[baseVec + idx] = uint4(as_uint2(olo), as_uint2(ohi));
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
