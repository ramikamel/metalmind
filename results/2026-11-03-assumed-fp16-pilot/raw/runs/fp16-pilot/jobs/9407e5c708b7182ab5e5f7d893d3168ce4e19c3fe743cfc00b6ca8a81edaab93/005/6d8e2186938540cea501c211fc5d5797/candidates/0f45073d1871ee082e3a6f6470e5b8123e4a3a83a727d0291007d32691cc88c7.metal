#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tptg [[threads_per_threadgroup]]) {
    (void)epsilon;
    if (row >= B) return;
    threadgroup float buf[256];
    device const half* in_row = x + (uint)row * D;
    device half* out_row = out + (uint)row * D;
    float local_max = -1.0e30f;
    for (uint i = tid; i < D; i += tptg) {
        local_max = fmax(local_max, float(in_row[i]));
    }
    buf[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tptg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) buf[tid] = fmax(buf[tid], buf[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = buf[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_sum = 0.0f;
    for (uint i = tid; i < D; i += tptg) {
        local_sum += exp(float(in_row[i]) - row_max);
    }
    buf[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tptg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) buf[tid] += buf[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / buf[0];
    for (uint i = tid; i < D; i += tptg) {
        float v = exp(float(in_row[i]) - row_max) * inv_sum;
        out_row[i] = half(v);
    }
}
