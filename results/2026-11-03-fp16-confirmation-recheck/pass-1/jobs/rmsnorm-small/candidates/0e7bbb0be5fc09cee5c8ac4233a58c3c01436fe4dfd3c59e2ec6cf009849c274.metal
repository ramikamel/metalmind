#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device const half* parameter, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint thread_id [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]]) {
    uint row = group_id;
    if (row >= B) return;
    threadgroup float partial[256];
    device const half* row_x = x + uint(row) * D;
    float sum = 0.0f;
    for (uint c = thread_id; c < D; c += 256u) {
        float v = float(row_x[c]);
        sum += v * v;
    }
    partial[thread_id] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (thread_id < stride) {
            partial[thread_id] += partial[thread_id + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float denom = partial[0] / float(D) + epsilon;
    float scale = rsqrt(denom);
    device half* row_out = out + uint(row) * D;
    for (uint c = thread_id; c < D; c += 256u) {
        float v = float(row_x[c]) * scale * float(parameter[c]);
        row_out[c] = half(v);
    }
}
