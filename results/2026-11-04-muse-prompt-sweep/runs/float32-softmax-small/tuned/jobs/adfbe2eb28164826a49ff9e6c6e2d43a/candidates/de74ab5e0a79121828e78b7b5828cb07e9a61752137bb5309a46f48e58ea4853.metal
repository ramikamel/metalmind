#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint width [[threads_per_simdgroup]]) {
    (void)epsilon;
    if (row >= B) return;
    uint base = row * D;
    float vals[16];
    uint count = 0;
    float local_max = -3.402823466e+38f;
    for (uint i = lane; i < D; i += width) {
        float v = x[base + i];
        if (count < 16) vals[count] = v;
        count++;
        local_max = max(local_max, v);
    }
    float row_max = simd_max(local_max);
    float local_sum = 0.0f;
    if (count <= 16) {
        for (uint k = 0; k < count; ++k) {
            float e = exp(vals[k] - row_max);
            vals[k] = e;
            local_sum += e;
        }
        float inv_sum = 1.0f / simd_sum(local_sum);
        for (uint k = 0; k < count; ++k) {
            uint i = lane + k * width;
            out[base + i] = vals[k] * inv_sum;
        }
    } else {
        for (uint i = lane; i < D; i += width) {
            local_sum += exp(x[base + i] - row_max);
        }
        float inv_sum = 1.0f / simd_sum(local_sum);
        for (uint i = lane; i < D; i += width) {
            out[base + i] = exp(x[base + i] - row_max) * inv_sum;
        }
    }
}
