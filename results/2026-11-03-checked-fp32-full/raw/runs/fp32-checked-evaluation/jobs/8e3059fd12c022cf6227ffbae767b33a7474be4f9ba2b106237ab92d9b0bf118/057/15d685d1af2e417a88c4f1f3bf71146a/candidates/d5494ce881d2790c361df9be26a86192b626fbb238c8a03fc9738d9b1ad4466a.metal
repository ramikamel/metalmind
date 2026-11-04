#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tg_pos [[threadgroup_position_in_grid]], uint3 tid_pos [[thread_position_in_threadgroup]], uint3 threads_per_tg [[threads_per_threadgroup]]) {
    const uint tpr = 32u;
    uint tid = tid_pos.x;
    uint rows_per_tg = threads_per_tg.x / tpr;
    uint row_in_tg = tid / tpr;
    uint lane = tid % tpr;
    uint row = tg_pos.x * rows_per_tg + row_in_tg;
    bool valid = row < B;
    uint base = row * D;
    float sum = 0.0f;
    if (valid) {
        for (uint c = lane; c < D; c += tpr) {
            float v = x[base + c];
            sum += v * v;
        }
    }
    threadgroup float partial[128];
    threadgroup float invs[4];
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        float s = 0.0f;
        uint pbase = row_in_tg * tpr;
        for (uint i = 0u; i < tpr; ++i) s += partial[pbase + i];
        invs[row_in_tg] = rsqrt(s / float(D) + epsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (valid) {
        float inv = invs[row_in_tg];
        for (uint c = lane; c < D; c += tpr) {
            out[base + c] = x[base + c] * inv * parameter[c];
        }
    }
}
