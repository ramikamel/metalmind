#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tg_pos [[threadgroup_position_in_grid]], uint3 tid_in_tg [[thread_position_in_threadgroup]], uint3 threads_per_tg [[threads_per_threadgroup]]) {
    (void)epsilon;
    (void)tg_pos;
    (void)threads_per_tg;
    uint row = tid_in_tg.x;
    if (row >= B) { return; }
    uint base = row * D;
    float row_max = -INFINITY;
    uint i = 0;
    for (; i + 4 <= D; i += 4) {
        float a = x[base + i];
        float b = x[base + i + 1];
        float c = x[base + i + 2];
        float d = x[base + i + 3];
        row_max = max(row_max, max(max(a, b), max(c, d)));
    }
    for (; i < D; ++i) { row_max = max(row_max, x[base + i]); }
    float row_sum = 0.0f;
    i = 0;
    for (; i + 4 <= D; i += 4) {
        float a = exp(x[base + i] - row_max);
        float b = exp(x[base + i + 1] - row_max);
        float c = exp(x[base + i + 2] - row_max);
        float d = exp(x[base + i + 3] - row_max);
        row_sum += (a + b) + (c + d);
    }
    for (; i < D; ++i) { row_sum += exp(x[base + i] - row_max); }
    float inv_sum = 1.0f / row_sum;
    i = 0;
    for (; i + 4 <= D; i += 4) {
        out[base + i] = exp(x[base + i] - row_max) * inv_sum;
        out[base + i + 1] = exp(x[base + i + 1] - row_max) * inv_sum;
        out[base + i + 2] = exp(x[base + i + 2] - row_max) * inv_sum;
        out[base + i + 3] = exp(x[base + i + 3] - row_max) * inv_sum;
    }
    for (; i < D; ++i) { out[base + i] = exp(x[base + i] - row_max) * inv_sum; }
}
