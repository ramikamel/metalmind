#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tg_pos [[threadgroup_position_in_grid]], uint3 tid_in_tg [[thread_position_in_threadgroup]], uint3 threads_per_tg [[threads_per_threadgroup]]) {
    (void)epsilon;
    uint row = tg_pos.x;
    if (row >= B) {
        return;
    }
    uint lane = tid_in_tg.x;
    uint width = threads_per_tg.x;
    uint base = row * D;
    threadgroup float shared[32];
    float local_max = -INFINITY;
    for (uint i = lane; i < D; i += width) {
        float v = x[base + i];
        local_max = max(local_max, v);
    }
    shared[lane] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        float m = shared[0];
        for (uint i = 1; i < width; ++i) {
            m = max(m, shared[i]);
        }
        shared[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = shared[0];
    float local_sum = 0.0f;
    for (uint i = lane; i < D; i += width) {
        local_sum += exp(x[base + i] - row_max);
    }
    shared[lane] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        float s = shared[0];
        for (uint i = 1; i < width; ++i) {
            s += shared[i];
        }
        shared[0] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = 1.0f / shared[0];
    for (uint i = lane; i < D; i += width) {
        float v = exp(x[base + i] - row_max) * inv_sum;
        out[base + i] = v;
    }
}
