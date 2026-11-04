#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 gid [[thread_position_in_grid]]) {
    uint row = gid.x;
    if (row >= B) return;
    uint base = row * D;
    float s0 = 0.0f;
    float s1 = 0.0f;
    float s2 = 0.0f;
    float s3 = 0.0f;
    uint i = 0;
    for (; i + 3u < D; i += 4u) {
        float v0 = x[base + i];
        float v1 = x[base + i + 1u];
        float v2 = x[base + i + 2u];
        float v3 = x[base + i + 3u];
        s0 += v0 * v0;
        s1 += v1 * v1;
        s2 += v2 * v2;
        s3 += v3 * v3;
    }
    for (; i < D; ++i) {
        float v = x[base + i];
        s0 += v * v;
    }
    float sum = (s0 + s1) + (s2 + s3);
    float scale = rsqrt(sum / float(D) + epsilon);
    for (uint j = 0; j < D; ++j) {
        out[base + j] = float(x[base + j] * scale * parameter[j]);
    }
}
