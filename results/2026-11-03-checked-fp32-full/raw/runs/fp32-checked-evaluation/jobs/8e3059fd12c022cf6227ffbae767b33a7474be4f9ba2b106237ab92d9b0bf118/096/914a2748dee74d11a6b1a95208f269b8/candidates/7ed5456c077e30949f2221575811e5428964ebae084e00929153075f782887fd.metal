#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tg_pos [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]], uint3 tg_size [[threads_per_threadgroup]]) {
    (void)epsilon;
    uint row = tg_pos.x;
    if (row >= B) return;
    uint nthreads = tg_size.x;
    uint base = row * D;
    threadgroup float cache[4096];
    threadgroup float red[256];
    float local_max = -3.402823466e+38f;
    for (uint i = lid; i < D; i += nthreads) {
        local_max = max(local_max, x[base + i]);
    }
    red[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = nthreads >> 1; s > 0; s >>= 1) {
        if (lid < s) red[lid] = max(red[lid], red[lid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool use_cache = D <= 4096;
    float local_sum = 0.0f;
    if (use_cache) {
        for (uint i = lid; i < D; i += nthreads) {
            float e = exp(x[base + i] - row_max);
            cache[i] = e;
            local_sum += e;
        }
    } else {
        for (uint i = lid; i < D; i += nthreads) {
            local_sum += exp(x[base + i] - row_max);
        }
    }
    red[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = nthreads >> 1; s > 0; s >>= 1) {
        if (lid < s) red[lid] += red[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / red[0];
    if (use_cache) {
        for (uint i = lid; i < D; i += nthreads) {
            out[base + i] = cache[i] * inv_sum;
        }
    } else {
        for (uint i = lid; i < D; i += nthreads) {
            out[base + i] = exp(x[base + i] - row_max) * inv_sum;
        }
    }
}
