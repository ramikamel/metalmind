#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device const half* parameter, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint2 gid [[thread_position_in_grid]]) {
    (void)epsilon;
    uint row = gid.y;
    uint col = gid.x * 8u;
    if (row >= B || col >= D) return;
    if (col + 8u <= D) {
        uint base4 = (row * D + col) >> 2u;
        uint pbase4 = col >> 2u;
        half4 xv0 = ((device const half4*)x)[base4];
        half4 xv1 = ((device const half4*)x)[base4 + 1u];
        half4 pv0 = ((device const half4*)parameter)[pbase4];
        half4 pv1 = ((device const half4*)parameter)[pbase4 + 1u];
        float4 z0 = float4(xv0) + float4(pv0);
        float4 z1 = float4(xv1) + float4(pv1);
        float4 a0 = 0.7978845608028654f * (z0 + 0.044715f * z0 * z0 * z0);
        float4 a1 = 0.7978845608028654f * (z1 + 0.044715f * z1 * z1 * z1);
        float4 y0 = 0.5f * z0 * (1.0f + tanh(a0));
        float4 y1 = 0.5f * z1 * (1.0f + tanh(a1));
        ((device half4*)out)[base4] = half4(y0);
        ((device half4*)out)[base4 + 1u] = half4(y1);
    } else {
        uint base = row * D + col;
        uint remaining = D - col;
        for (uint i = 0u; i < remaining; ++i) {
            float z = float(x[base + i]) + float(parameter[col + i]);
            float a = 0.7978845608028654f * (z + 0.044715f * z * z * z);
            float y = 0.5f * z * (1.0f + tanh(a));
            out[base + i] = half(y);
        }
    }
}
