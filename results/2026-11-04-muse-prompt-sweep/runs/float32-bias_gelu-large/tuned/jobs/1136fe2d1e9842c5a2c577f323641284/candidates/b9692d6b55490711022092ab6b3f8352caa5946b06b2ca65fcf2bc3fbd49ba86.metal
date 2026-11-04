#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 gid [[thread_position_in_grid]]) {
    (void)epsilon;
    uint colBase = gid.x * 8u;
    uint rowBase = gid.y * 8u;
    if (colBase >= D || rowBase >= B) return;
    if ((D & 7u) == 0u) {
        float4 bias0 = ((device const float4*)(parameter + colBase))[0];
        float4 bias1 = ((device const float4*)(parameter + colBase))[1];
        for (uint r = 0u; r < 8u; ++r) {
            uint row = rowBase + r;
            if (row >= B) break;
            uint offset = row * D + colBase;
            float4 xv0 = ((device const float4*)(x + offset))[0];
            float4 xv1 = ((device const float4*)(x + offset))[1];
            float4 z0 = xv0 + bias0;
            float4 z1 = xv1 + bias1;
            float4 z20 = z0 * z0;
            float4 z21 = z1 * z1;
            float4 u0 = 0.7978845608028654f * z0 * (1.0f + 0.044715f * z20);
            float4 u1 = 0.7978845608028654f * z1 * (1.0f + 0.044715f * z21);
            float4 y0 = 0.5f * z0 * (1.0f + tanh(u0));
            float4 y1 = 0.5f * z1 * (1.0f + tanh(u1));
            ((device float4*)(out + offset))[0] = y0;
            ((device float4*)(out + offset))[1] = y1;
        }
        return;
    }
    for (uint r = 0u; r < 8u; ++r) {
        uint row = rowBase + r;
        if (row >= B) break;
        uint base = row * D + colBase;
        for (uint j = 0u; j < 8u; ++j) {
            uint col = colBase + j;
            if (col >= D) break;
            float z = x[base + j] + parameter[col];
            float z2 = z * z;
            float u = 0.7978845608028654f * z * (1.0f + 0.044715f * z2);
            float y = 0.5f * z * (1.0f + tanh(u));
            out[base + j] = y;
        }
    }
}
