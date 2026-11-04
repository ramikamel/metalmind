#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint tid [[thread_position_in_threadgroup]], uint row [[threadgroup_position_in_grid]], uint gtid [[thread_position_in_grid]]) {
    (void)gtid;
    (void)epsilon;
    if (row >= B) { return; }
    threadgroup float sdata[64];
    threadgroup float cache[257];
    uint base = row * D;
    float local_max = -3.402823466e+38f;
    for (uint i = tid; i < D; i += 64u) {
        local_max = max(local_max, float(x[base + i]));
    }
    sdata[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float m = sdata[0];
        for (uint k = 1u; k < 64u; ++k) { m = max(m, sdata[k]); }
        sdata[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = sdata[0];
    float local_sum = 0.0f;
    for (uint i = tid; i < D; i += 64u) {
        float e = exp(float(x[base + i]) - row_max);
        cache[i] = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float s = sdata[0];
        for (uint k = 1u; k < 64u; ++k) { s += sdata[k]; }
        sdata[0] = 1.0f / s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = sdata[0];
    for (uint i = tid; i < D; i += 64u) {
        out[base + i] = half(cache[i] * inv_sum);
    }
}
