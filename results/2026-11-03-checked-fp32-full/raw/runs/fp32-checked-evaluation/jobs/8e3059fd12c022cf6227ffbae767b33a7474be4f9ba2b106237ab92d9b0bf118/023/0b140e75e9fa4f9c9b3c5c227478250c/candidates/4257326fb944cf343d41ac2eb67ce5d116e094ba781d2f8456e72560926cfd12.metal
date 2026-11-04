#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tgid [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 tgdim [[threads_per_threadgroup]]) {
    uint row = tgid.x;
    if (row >= B) return;
    uint lane = tid.x;
    uint nthreads = tgdim.x;
    device const float* row_x = x + row * D;
    device float* row_o = out + row * D;
    threadgroup float shared[256];
    float local_max = -3.402823466e+38f;
    for (uint i = lane; i < D; i += nthreads) {
        local_max = max(local_max, row_x[i]);
    }
    shared[lane] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = nthreads >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) shared[lane] = max(shared[lane], shared[lane + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_sum = 0.0f;
    for (uint i = lane; i < D; i += nthreads) {
        local_sum += exp(row_x[i] - row_max);
    }
    shared[lane] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = nthreads >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) shared[lane] += shared[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / shared[0];
    for (uint i = lane; i < D; i += nthreads) {
        row_o[i] = float(exp(row_x[i] - row_max) * inv_sum);
    }
}
