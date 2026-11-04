#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint width [[threads_per_simdgroup]]) {
    (void)epsilon;
    if (row >= B) return;
    uint base = row * D;
    if (D > 512u) {
        float local_max = -3.402823466e+38f;
        for (uint i = lane; i < D; i += width) local_max = max(local_max, x[base + i]);
        float row_max = simd_max(local_max);
        float local_sum = 0.0f;
        for (uint i = lane; i < D; i += width) {
            float e = exp2((x[base + i] - row_max) * 1.4426950408889634f);
            out[base + i] = e;
            local_sum += e;
        }
        float inv_sum = 1.0f / simd_sum(local_sum);
        for (uint i = lane; i < D; i += width) out[base + i] = out[base + i] * inv_sum;
        return;
    }
    threadgroup float cache[512];
    float local_max = -3.402823466e+38f;
    for (uint i = lane; i < D; i += width) {
        float v = x[base + i];
        cache[i] = v;
        local_max = max(local_max, v);
    }
    float row_max = simd_max(local_max);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    float local_sum = 0.0f;
    for (uint i = lane; i < D; i += width) {
        float e = exp2((cache[i] - row_max) * 1.4426950408889634f);
        cache[i] = e;
        local_sum += e;
    }
    float inv_sum = 1.0f / simd_sum(local_sum);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lane; i < D; i += width) out[base + i] = cache[i] * inv_sum;
}
