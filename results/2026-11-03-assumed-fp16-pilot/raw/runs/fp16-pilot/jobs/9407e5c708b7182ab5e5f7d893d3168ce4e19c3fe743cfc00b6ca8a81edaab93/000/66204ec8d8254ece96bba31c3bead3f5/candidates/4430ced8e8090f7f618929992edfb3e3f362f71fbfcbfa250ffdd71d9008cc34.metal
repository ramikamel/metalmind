#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint tid [[thread_position_in_threadgroup]], uint row [[threadgroup_position_in_grid]], uint gtid [[thread_position_in_grid]]) {
    (void)gtid;
    (void)epsilon;
    if (row >= B) { return; }
    threadgroup float sdata[256];
    threadgroup float cache[257];
    uint base = row * D;
    float local_max = -3.402823466e+38f;
    for (uint i = tid; i < D; i += 256u) {
        float v = float(x[base + i]);
        local_max = max(local_max, v);
    }
    sdata[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) { sdata[tid] = max(sdata[tid], sdata[tid + s]); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = sdata[0];
    float local_sum = 0.0f;
    for (uint i = tid; i < D; i += 256u) {
        float e = exp(float(x[base + i]) - row_max);
        cache[i] = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) { sdata[tid] += sdata[tid + s]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / sdata[0];
    for (uint i = tid; i < D; i += 256u) {
        out[base + i] = half(cache[i] * inv_sum);
    }
}
