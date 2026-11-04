#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device const half* parameter, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tg_pos [[threadgroup_position_in_grid]], uint3 tp_tg [[thread_position_in_threadgroup]], uint3 tptg [[threads_per_threadgroup]], uint lid [[thread_index_in_simdgroup]], uint sgid [[simdgroup_index_in_threadgroup]], uint sptg [[simdgroups_per_threadgroup]]) {
    threadgroup float sg_sums[64];
    threadgroup float tg_scale;
    uint row = tg_pos.x;
    if (row >= B) {
        return;
    }
    uint tid = tp_tg.x;
    uint nt = tptg.x;
    bool fast_path = (D == 1024u && nt == 256u);
    float4 cached = float4(0.0f);
    float sum = 0.0f;
    if (fast_path) {
        cached = float4(((device const half4*)x)[row * 256u + tid]);
        sum = dot(cached, cached);
    } else if ((D & 3u) == 0u) {
        uint nv = D >> 2;
        for (uint i = tid; i < nv; i += nt) {
            float4 v = float4(((device const half4*)x)[row * nv + i]);
            sum += dot(v, v);
        }
    } else {
        for (uint i = tid; i < D; i += nt) {
            float v = float(x[row * D + i]);
            sum += v * v;
        }
    }
    float sg_sum = simd_sum(sum);
    if (lid == 0u) {
        sg_sums[sgid] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        float s = (lid < sptg) ? sg_sums[lid] : 0.0f;
        s = simd_sum(s);
        if (lid == 0u) {
            tg_scale = rsqrt(s / float(D) + epsilon);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = tg_scale;
    if (fast_path) {
        float4 w = float4(((device const half4*)parameter)[tid]);
        ((device half4*)out)[row * 256u + tid] = half4(cached * scale * w);
    } else if ((D & 3u) == 0u) {
        uint nv = D >> 2;
        for (uint i = tid; i < nv; i += nt) {
            float4 v = float4(((device const half4*)x)[row * nv + i]);
            float4 w = float4(((device const half4*)parameter)[i]);
            ((device half4*)out)[row * nv + i] = half4(v * scale * w);
        }
    } else {
        for (uint i = tid; i < D; i += nt) {
            float v = float(x[row * D + i]);
            float w = float(parameter[i]);
            out[row * D + i] = half(v * scale * w);
        }
    }
}
