#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint gid [[thread_position_in_grid]], uint total_threads [[threads_per_grid]]) {
    (void)epsilon;
    if (D == 0u) return;
    uint chunks_per_row = (D + 3u) / 4u;
    uint total_chunks = B * chunks_per_row;
    for (uint chunk = gid; chunk < total_chunks; chunk += total_threads) {
        uint row = chunk / chunks_per_row;
        uint col = (chunk - row * chunks_per_row) * 4u;
        uint base = row * D + col;
        if (col + 3u < D) {
            float4 z = float4(x[base], x[base + 1u], x[base + 2u], x[base + 3u]) + float4(parameter[col], parameter[col + 1u], parameter[col + 2u], parameter[col + 3u]);
            float4 inner = 0.7978845608028654f * (z + 0.044715f * z * z * z);
            float4 y = 0.5f * z * (1.0f + tanh(inner));
            out[base] = y.x;
            out[base + 1u] = y.y;
            out[base + 2u] = y.z;
            out[base + 3u] = y.w;
        } else {
            for (uint c = col; c < D; ++c) {
                uint idx = row * D + c;
                float z = x[idx] + parameter[c];
                float inner = 0.7978845608028654f * (z + 0.044715f * z * z * z);
                out[idx] = 0.5f * z * (1.0f + tanh(inner));
            }
        }
    }
}
