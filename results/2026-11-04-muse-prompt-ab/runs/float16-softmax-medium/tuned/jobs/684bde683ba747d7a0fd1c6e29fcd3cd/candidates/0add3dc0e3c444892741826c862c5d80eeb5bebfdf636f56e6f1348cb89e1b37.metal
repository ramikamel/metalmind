#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint W [[threads_per_simdgroup]]) {
    (void)epsilon;
    if (row >= B) return;
    uint base = row * D;
    if (D <= 32u * W) {
        float vals[32];
        uint n = 0u;
        float local_max = -3.4028235e38f;
        for (uint j = lane; j < D; j += W) {
            float v = float(x[base + j]);
            vals[n] = v;
            local_max = max(local_max, v);
            n += 1u;
        }
        float row_max = simd_max(local_max);
        float local_sum = 0.0f;
        for (uint i = 0u; i < n; i += 1u) {
            float e = exp(vals[i] - row_max);
            vals[i] = e;
            local_sum += e;
        }
        float row_sum = simd_sum(local_sum);
        float inv = row_sum > 0.0f ? 1.0f / row_sum : 0.0f;
        uint idx = 0u;
        for (uint j = lane; j < D; j += W) {
            out[base + j] = half(vals[idx] * inv);
            idx += 1u;
        }
    } else {
        float local_max = -3.4028235e38f;
        for (uint j = lane; j < D; j += W) {
            local_max = max(local_max, float(x[base + j]));
        }
        float row_max = simd_max(local_max);
        float local_sum = 0.0f;
        for (uint j = lane; j < D; j += W) {
            local_sum += exp(float(x[base + j]) - row_max);
        }
        float row_sum = simd_sum(local_sum);
        float inv = row_sum > 0.0f ? 1.0f / row_sum : 0.0f;
        for (uint j = lane; j < D; j += W) {
            float e = exp(float(x[base + j]) - row_max);
            out[base + j] = half(e * inv);
        }
    }
}
