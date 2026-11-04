#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device const half* parameter, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint2 gid [[thread_position_in_grid]]) {
    (void)epsilon;
    uint row = gid.y;
    uint col = gid.x << 2u;
    if (row >= B || col >= D) {
        return;
    }
    uint base = row * D + col;
    uint rem = D - col;
    if (rem >= 4u && ((D & 3u) == 0u)) {
        half4 xv = ((device const half4*)(x + base))[0];
        half4 pv = ((device const half4*)(parameter + col))[0];
        float4 z = float4(xv) + float4(pv);
        float4 z2 = z * z;
        float4 inner = 0.79788456f * (z + 0.044715f * z2 * z);
        float4 y = 0.5f * z * (1.0f + tanh(inner));
        ((device half4*)(out + base))[0] = half4(y);
    } else {
        uint n = rem < 4u ? rem : 4u;
        for (uint i = 0u; i < n; ++i) {
            float z = float(x[base + i]) + float(parameter[col + i]);
            float cube = z * z * z;
            float inner = 0.79788456f * (z + 0.044715f * cube);
            float y = 0.5f * z * (1.0f + tanh(inner));
            out[base + i] = half(y);
        }
    }
}
