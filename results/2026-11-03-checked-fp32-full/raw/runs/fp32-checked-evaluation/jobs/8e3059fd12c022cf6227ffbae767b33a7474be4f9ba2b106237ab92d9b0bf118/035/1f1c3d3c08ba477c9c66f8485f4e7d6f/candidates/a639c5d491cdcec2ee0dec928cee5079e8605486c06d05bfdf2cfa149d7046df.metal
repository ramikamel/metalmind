#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint lid [[thread_position_in_threadgroup]], uint gid [[threadgroup_position_in_grid]], uint tgs [[threads_per_threadgroup]]) {
    threadgroup float shared_data[256];
    uint row = gid;
    if (row >= B) {
        return;
    }
    uint base = row * D;
    float local_max = -3.4028235e38f;
    for (uint i = lid; i < D; i += tgs) {
        float v = x[base + i];
        local_max = max(local_max, v);
    }
    shared_data[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgs / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared_data[lid] = max(shared_data[lid], shared_data[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared_data[0];
    float local_sum = 0.0f;
    for (uint i = lid; i < D; i += tgs) {
        float v = x[base + i];
        local_sum += exp(v - row_max);
    }
    shared_data[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgs / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared_data[lid] += shared_data[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / shared_data[0];
    for (uint i = lid; i < D; i += tgs) {
        float v = x[base + i];
        out[base + i] = float(exp(v - row_max) * inv_sum);
    }
}
