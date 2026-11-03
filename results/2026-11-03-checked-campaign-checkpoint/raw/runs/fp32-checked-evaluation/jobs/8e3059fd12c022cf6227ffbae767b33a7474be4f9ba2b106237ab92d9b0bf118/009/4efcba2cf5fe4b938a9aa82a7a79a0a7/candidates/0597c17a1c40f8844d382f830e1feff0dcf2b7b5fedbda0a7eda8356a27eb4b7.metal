#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 gid [[thread_position_in_grid]]) {
    (void)epsilon;
    uint row = gid.x;
    if (row >= B) {
        return;
    }
    uint base = row * D;
    float row_max = -3.4028234663852886e+38f;
    for (uint i = 0; i < D; ++i) {
        float v = x[base + i];
        row_max = v > row_max ? v : row_max;
    }
    float row_sum = 0.0f;
    for (uint i = 0; i < D; ++i) {
        float e = exp(x[base + i] - row_max);
        out[base + i] = e;
        row_sum += e;
    }
    float inv_sum = 1.0f / row_sum;
    for (uint i = 0; i < D; ++i) {
        out[base + i] *= inv_sum;
    }
}
