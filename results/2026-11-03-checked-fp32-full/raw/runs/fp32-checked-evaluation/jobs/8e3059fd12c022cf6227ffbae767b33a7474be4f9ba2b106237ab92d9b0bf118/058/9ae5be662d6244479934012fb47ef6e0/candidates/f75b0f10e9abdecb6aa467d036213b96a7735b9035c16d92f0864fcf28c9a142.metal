#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint2 pos [[thread_position_in_grid]], uint2 grid [[threads_per_grid]]) {
    (void)epsilon;
    const float k = 0.7978845608028654f;
    const uint colVecs = (D + 3u) / 4u;
    for (uint row = pos.y; row < B; row += grid.y) {
        const uint rowBase = row * D;
        for (uint cv = pos.x; cv < colVecs; cv += grid.x) {
            const uint col = cv * 4u;
            const uint idx = rowBase + col;
            if (col + 3u < D) {
                float4 xv = *(device const float4*)(x + idx);
                float4 pv = *(device const float4*)(parameter + col);
                float4 z = xv + pv;
                float4 z2 = z * z;
                float4 inner = k * (z + 0.044715f * z2 * z);
                float4 y;
                y.x = 0.5f * z.x * (1.0f + tanh(inner.x));
                y.y = 0.5f * z.y * (1.0f + tanh(inner.y));
                y.z = 0.5f * z.z * (1.0f + tanh(inner.z));
                y.w = 0.5f * z.w * (1.0f + tanh(inner.w));
                *(device float4*)(out + idx) = y;
            } else {
                for (uint j = 0u; j < 4u; ++j) {
                    const uint c = col + j;
                    if (c < D) {
                        float z = x[idx + j] + parameter[c];
                        float inner = k * (z + 0.044715f * z * z * z);
                        out[idx + j] = 0.5f * z * (1.0f + tanh(inner));
                    }
                }
            }
        }
    }
}
