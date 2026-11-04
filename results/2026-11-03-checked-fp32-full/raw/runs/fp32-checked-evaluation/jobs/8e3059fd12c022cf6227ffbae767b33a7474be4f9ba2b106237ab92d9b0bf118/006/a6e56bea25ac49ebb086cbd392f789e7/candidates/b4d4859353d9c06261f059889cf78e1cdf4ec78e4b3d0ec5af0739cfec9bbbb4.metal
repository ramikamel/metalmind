#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint row [[thread_position_in_grid]]) {
    if (row >= B || D == 0u) {
        return;
    }
    uint base = row * D;
    float sum = 0.0f;
    uint j = 0u;
    uint d4 = D - (D % 4u);
    for (; j < d4; j += 4u) {
        float a = x[base + j];
        float b = x[base + j + 1u];
        float c = x[base + j + 2u];
        float d = x[base + j + 3u];
        sum += a * a + b * b + c * c + d * d;
    }
    for (; j < D; ++j) {
        float v = x[base + j];
        sum += v * v;
    }
    float inv = rsqrt(sum / float(D) + epsilon);
    j = 0u;
    for (; j < d4; j += 4u) {
        out[base + j] = x[base + j] * inv * parameter[j];
        out[base + j + 1u] = x[base + j + 1u] * inv * parameter[j + 1u];
        out[base + j + 2u] = x[base + j + 2u] * inv * parameter[j + 2u];
        out[base + j + 3u] = x[base + j + 3u] * inv * parameter[j + 3u];
    }
    for (; j < D; ++j) {
        out[base + j] = x[base + j] * inv * parameter[j];
    }
}
