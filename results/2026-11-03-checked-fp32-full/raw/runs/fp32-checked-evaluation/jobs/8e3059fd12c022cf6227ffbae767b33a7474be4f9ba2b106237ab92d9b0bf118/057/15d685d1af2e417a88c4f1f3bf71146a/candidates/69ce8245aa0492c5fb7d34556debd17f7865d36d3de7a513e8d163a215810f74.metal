#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tg_pos [[threadgroup_position_in_grid]], uint3 tid_pos [[thread_position_in_threadgroup]], uint3 threads_per_tg [[threads_per_threadgroup]]) {
    uint row = tg_pos.x;
    if (row >= B) return;
    uint tid = tid_pos.x;
    uint nthreads = threads_per_tg.x;
    uint base = row * D;
    threadgroup float partial[128];
    threadgroup float cache[257];
    float sum = 0.0f;
    for (uint c = tid; c < D; c += nthreads) {
        float v = x[base + c];
        cache[c] = v;
        sum += v * v;
    }
    if (tid < 128u) partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = nthreads >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(partial[0] / float(D) + epsilon);
    for (uint c = tid; c < D; c += nthreads) {
        out[base + c] = cache[c] * inv * parameter[c];
    }
}
