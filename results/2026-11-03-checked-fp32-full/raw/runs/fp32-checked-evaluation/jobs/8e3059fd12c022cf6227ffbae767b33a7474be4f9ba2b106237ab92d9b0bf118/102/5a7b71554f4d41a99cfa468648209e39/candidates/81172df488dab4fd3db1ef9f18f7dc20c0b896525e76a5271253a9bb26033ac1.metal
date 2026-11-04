#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tgpos [[threadgroup_position_in_grid]], uint3 tpos [[thread_position_in_threadgroup]], uint3 tgsize [[threads_per_threadgroup]]) {
    (void)epsilon;
    threadgroup float shared[256];
    uint row = tgpos.x;
    uint tid = tpos.x;
    uint nt = tgsize.x;
    bool valid = row < B;
    float local_max = -1.0e38f;
    if (valid) {
        for (uint i = tid; i < D; i += nt) {
            float v = x[(uint)row * D + i];
            local_max = max(local_max, v);
        }
    }
    shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s && tid + s < nt) {
            shared[tid] = max(shared[tid], shared[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_sum = 0.0f;
    if (valid) {
        for (uint i = tid; i < D; i += nt) {
            float v = x[(uint)row * D + i];
            local_sum += exp(v - row_max);
        }
    }
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s && tid + s < nt) {
            shared[tid] = shared[tid] + shared[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_sum = shared[0];
    float inv = 1.0f / row_sum;
    if (valid) {
        for (uint i = tid; i < D; i += nt) {
            float v = x[(uint)row * D + i];
            float e = exp(v - row_max);
            out[(uint)row * D + i] = e * inv;
        }
    }
}
