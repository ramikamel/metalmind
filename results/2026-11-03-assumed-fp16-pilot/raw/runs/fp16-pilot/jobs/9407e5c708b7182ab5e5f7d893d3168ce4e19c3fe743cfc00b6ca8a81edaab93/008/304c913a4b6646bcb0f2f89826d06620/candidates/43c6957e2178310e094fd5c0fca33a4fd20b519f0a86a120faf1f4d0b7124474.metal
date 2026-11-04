#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device const half* parameter, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint tid [[thread_index_in_threadgroup]], uint group_id [[threadgroup_position_in_grid]], uint simd_size [[threads_per_simdgroup]]) {
    uint row = group_id;
    if (row >= B) return;
    device const half* row_x = x + row * D;
    float sum = 0.0f;
    for (uint c = tid; c < D; c += simd_size) {
        float v = float(row_x[c]);
        sum += v * v;
    }
    float scale = rsqrt(simd_sum(sum) / float(D) + epsilon);
    device half* row_out = out + row * D;
    for (uint c = tid; c < D; c += simd_size) {
        row_out[c] = half(float(row_x[c]) * scale * float(parameter[c]));
    }
}
