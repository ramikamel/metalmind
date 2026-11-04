#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tgpos [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint simd_width [[threads_per_simdgroup]]) {
    (void)epsilon;
    uint row = tgpos.x;
    if (row >= B) return;
    uint base = row * D;
    uint width = simd_width == 0u ? 1u : simd_width;
    uint iters = (D + width - 1u) / width;
    if (iters <= 9u) {
        float vals[9];
        float local_max = -3.4028235e38f;
        for (uint k = 0u; k < 9u; ++k) {
            uint col = lane + k * width;
            float v = -3.4028235e38f;
            if (col < D) v = float(x[base + col]);
            vals[k] = v;
            local_max = max(local_max, v);
        }
        float row_max = simd_max(local_max);
        float local_sum = 0.0f;
        for (uint k = 0u; k < 9u; ++k) {
            uint col = lane + k * width;
            float e = 0.0f;
            if (col < D) e = exp(vals[k] - row_max);
            vals[k] = e;
            local_sum += e;
        }
        float row_sum = simd_sum(local_sum);
        for (uint k = 0u; k < 9u; ++k) {
            uint col = lane + k * width;
            if (col < D) out[base + col] = half(vals[k] / row_sum);
        }
    } else {
        float local_max = -3.4028235e38f;
        for (uint col = lane; col < D; col += width) local_max = max(local_max, float(x[base + col]));
        float row_max = simd_max(local_max);
        float local_sum = 0.0f;
        for (uint col = lane; col < D; col += width) local_sum += exp(float(x[base + col]) - row_max);
        float row_sum = simd_sum(local_sum);
        for (uint col = lane; col < D; col += width) out[base + col] = half(exp(float(x[base + col]) - row_max) / row_sum);
    }
}
