#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint gid [[thread_position_in_grid]], uint row [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]], uint tgsize [[threads_per_threadgroup]]) {
    (void)gid;
    (void)epsilon;
    if (row >= B) return;
    if (D == 0u) return;
    threadgroup float red[256];
    uint base = row * D;
    float local_max = -3.4028234663852886e+38f;
    for (uint i = lid; i < D; i += tgsize) {
        local_max = max(local_max, x[base + i]);
    }
    red[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgsize >> 1u; stride > 0u; stride >>= 1u) {
        if (lid < stride) red[lid] = max(red[lid], red[lid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_sum = 0.0f;
    for (uint i = lid; i < D; i += tgsize) {
        float v = exp(x[base + i] - row_max);
        out[base + i] = v;
        local_sum += v;
    }
    red[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgsize >> 1u; stride > 0u; stride >>= 1u) {
        if (lid < stride) red[lid] += red[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / red[0];
    for (uint i = lid; i < D; i += tgsize) {
        out[base + i] = out[base + i] * inv_sum;
    }
}
