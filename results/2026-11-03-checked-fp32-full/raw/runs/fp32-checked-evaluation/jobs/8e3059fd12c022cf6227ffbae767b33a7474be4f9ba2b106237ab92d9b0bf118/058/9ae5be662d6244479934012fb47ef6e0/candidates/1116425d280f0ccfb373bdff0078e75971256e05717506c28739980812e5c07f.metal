#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint2 pos [[thread_position_in_grid]]) {
    (void)epsilon;
    const uint row = pos.y;
    const uint cv = pos.x;
    const uint col = cv * 4u;
    if (row >= B || col >= D) return;
    const uint idx = row * D + col;
    if (col + 3u < D) {
        float4 z = *(device const float4*)(x + idx) + *(device const float4*)(parameter + col);
        float4 inner = 0.7978845608028654f * (z + 0.044715f * z * z * z);
        float4 y = 0.5f * z * (1.0f + tanh(inner));
        *(device float4*)(out + idx) = y;
    } else {
        for (uint j = 0u; j < 4u; ++j) {
            const uint c = col + j;
            if (c < D) {
                float z = x[idx + j] + parameter[c];
                float inner = 0.7978845608028654f * (z + 0.044715f * z * z * z);
                out[idx + j] = 0.5f * z * (1.0f + tanh(inner));
            }
        }
    }
}
