#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tg [[threadgroup_position_in_grid]], uint3 tpig [[thread_position_in_threadgroup]], uint3 tpg [[threads_per_threadgroup]]) {
    (void)epsilon;
    uint row = tg.x;
    if (row >= B) {
        return;
    }
    uint lid = tpig.x;
    uint nthreads = tpg.x;
    threadgroup float red[256];
    float local_max = -3.4028234663852886e+38f;
    for (uint i = lid; i < D; i += nthreads) {
        float v = x[row * D + i];
        local_max = v > local_max ? v : local_max;
    }
    red[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            float other = red[lid + stride];
            red[lid] = other > red[lid] ? other : red[lid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_sum = 0.0f;
    for (uint i = lid; i < D; i += nthreads) {
        local_sum += exp(x[row * D + i] - row_max);
    }
    red[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            red[lid] += red[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_sum = red[0];
    float inv_sum = 1.0f / row_sum;
    for (uint i = lid; i < D; i += nthreads) {
        uint idx = row * D + i;
        out[idx] = exp(x[idx] - row_max) * inv_sum;
    }
}
