#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 gid [[thread_position_in_grid]]) {
    (void)epsilon;
    uint row = gid.y;
    uint col = gid.x * 4u;
    if (row >= B || col >= D) {
        return;
    }
    uint base = row * D + col;
    if ((D & 3u) == 0u) {
        float4 xv = reinterpret_cast<device const float4*>(x + base)[0];
        float4 pv = reinterpret_cast<device const float4*>(parameter + col)[0];
        float4 z = xv + pv;
        float4 z3 = z * z * z;
        float4 inner = 0.7978845608028654f * (z + 0.044715f * z3);
        float4 y = 0.5f * z * (1.0f + tanh(inner));
        reinterpret_cast<device float4*>(out + base)[0] = y;
    } else {
        for (uint i = 0u; i < 4u; ++i) {
            uint c = col + i;
            if (c < D) {
                float z = x[base + i] + parameter[c];
                float z3 = z * z * z;
                float inner = 0.7978845608028654f * (z + 0.044715f * z3);
                out[base + i] = 0.5f * z * (1.0f + tanh(inner));
            }
        }
    }
}
