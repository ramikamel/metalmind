#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 gid [[thread_position_in_grid]]) {
    (void)epsilon;
    uint colBlock = gid.x;
    uint rowBase = gid.y * 4u;
    uint colBase = colBlock * 4u;
    if (colBase >= D || rowBase >= B) return;
    if ((D & 3u) != 0u) {
        for (uint r = 0u; r < 4u; ++r) {
            uint row = rowBase + r;
            if (row >= B) break;
            uint base = row * D + colBase;
            for (uint j = 0u; j < 4u; ++j) {
                uint col = colBase + j;
                if (col >= D) break;
                float z = x[base + j] + parameter[col];
                float z2 = z * z;
                float u = 0.7978845608028654f * z * (1.0f + 0.044715f * z2);
                float y = 0.5f * z * (1.0f + tanh(u));
                out[base + j] = y;
            }
        }
        return;
    }
    float4 bias = ((device const float4*)(parameter))[colBlock];
    for (uint r = 0u; r < 4u; ++r) {
        uint row = rowBase + r;
        if (row >= B) break;
        uint offset = row * D + colBase;
        float4 xv = ((device const float4*)(x + offset))[0];
        float4 z = xv + bias;
        float4 z2 = z * z;
        float4 u = 0.7978845608028654f * z * (1.0f + 0.044715f * z2);
        float4 y = 0.5f * z * (1.0f + tanh(u));
        ((device float4*)(out + offset))[0] = y;
    }
}
