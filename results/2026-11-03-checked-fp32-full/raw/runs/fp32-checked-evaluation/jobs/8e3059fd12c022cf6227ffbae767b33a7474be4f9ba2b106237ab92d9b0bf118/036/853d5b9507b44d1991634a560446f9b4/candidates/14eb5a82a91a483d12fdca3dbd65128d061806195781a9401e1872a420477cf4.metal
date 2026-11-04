#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint2 gid [[thread_position_in_grid]]) {
    (void)epsilon;
    uint cols4 = (D + 3u) / 4u;
    if (gid.y >= B || gid.x >= cols4) {
        return;
    }
    if ((D & 3u) == 0u) {
        device const float4* x4 = reinterpret_cast<device const float4*>(x);
        device const float4* p4 = reinterpret_cast<device const float4*>(parameter);
        device float4* o4 = reinterpret_cast<device float4*>(out);
        float4 z = x4[gid.y * cols4 + gid.x] + p4[gid.x];
        float4 cubic = z * z * z;
        float4 inner = 0.7978845608028654f * (z + 0.044715f * cubic);
        float4 y = 0.5f * z * (1.0f + tanh(inner));
        o4[gid.y * cols4 + gid.x] = y;
    } else {
        uint col = gid.x * 4u;
        uint base = gid.y * D;
        for (uint i = 0u; i < 4u; ++i) {
            uint c = col + i;
            if (c < D) {
                float z = x[base + c] + parameter[c];
                float inner = 0.7978845608028654f * (z + 0.044715f * z * z * z);
                out[base + c] = 0.5f * z * (1.0f + tanh(inner));
            }
        }
    }
}
