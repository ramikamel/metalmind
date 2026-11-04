#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint2 gid [[thread_position_in_grid]]) {
    (void)epsilon;
    uint colBlocks = (D + 3u) >> 2;
    if (gid.x >= colBlocks || gid.y >= B) {
        return;
    }
    uint col = gid.x << 2;
    uint idx = gid.y * D + col;
    const float k = 0.7978845608028654f;
    if (col + 3u < D) {
        float4 z = float4(x[idx], x[idx + 1u], x[idx + 2u], x[idx + 3u]) + float4(parameter[col], parameter[col + 1u], parameter[col + 2u], parameter[col + 3u]);
        float4 z3 = z * z * z;
        float4 y = 0.5f * z * (1.0f + tanh(k * (z + 0.044715f * z3)));
        out[idx] = y.x;
        out[idx + 1u] = y.y;
        out[idx + 2u] = y.z;
        out[idx + 3u] = y.w;
    } else {
        for (uint i = 0u; i < 4u && col + i < D; ++i) {
            float z = x[idx + i] + parameter[col + i];
            float z3 = z * z * z;
            out[idx + i] = 0.5f * z * (1.0f + tanh(k * (z + 0.044715f * z3)));
        }
    }
}
