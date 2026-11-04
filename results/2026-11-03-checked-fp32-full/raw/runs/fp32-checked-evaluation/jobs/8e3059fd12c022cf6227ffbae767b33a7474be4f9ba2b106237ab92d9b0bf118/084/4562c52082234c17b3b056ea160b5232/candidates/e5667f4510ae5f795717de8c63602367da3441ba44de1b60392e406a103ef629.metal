#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint tgdim [[threads_per_threadgroup]]) {
    if (tgid >= B) {
        return;
    }
    threadgroup float red[256];
    uint base = tgid * D;
    float local_max = -3.4028234663852886e+38f;
    for (uint i = tid; i < D; i += tgdim) {
        local_max = max(local_max, x[base + i]);
    }
    red[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            red[tid] = max(red[tid], red[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_sum = 0.0f;
    for (uint i = tid; i < D; i += tgdim) {
        float e = exp(x[base + i] - row_max);
        out[base + i] = e;
        local_sum += e;
    }
    red[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            red[tid] += red[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / red[0];
    for (uint i = tid; i < D; i += tgdim) {
        out[base + i] = out[base + i] * inv_sum;
    }
}
