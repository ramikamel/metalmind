#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device const half* parameter, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint2 gid [[thread_position_in_grid]]) {
    (void)epsilon;
    uint col = gid.x * 16u;
    uint row0 = gid.y * 4u;
    if (row0 >= B || col >= D) return;
    if (col + 16u <= D && row0 + 4u <= B) {
        uint pbase4 = col >> 2u;
        float4 p0 = float4(((device const half4*)parameter)[pbase4]);
        float4 p1 = float4(((device const half4*)parameter)[pbase4 + 1u]);
        float4 p2 = float4(((device const half4*)parameter)[pbase4 + 2u]);
        float4 p3 = float4(((device const half4*)parameter)[pbase4 + 3u]);
        for (uint r = 0u; r < 4u; ++r) {
            uint row = row0 + r;
            uint base4 = (row * D + col) >> 2u;
            float4 z0 = float4(((device const half4*)x)[base4]) + p0;
            float4 z1 = float4(((device const half4*)x)[base4 + 1u]) + p1;
            float4 z2 = float4(((device const half4*)x)[base4 + 2u]) + p2;
            float4 z3 = float4(((device const half4*)x)[base4 + 3u]) + p3;
            float4 a0 = 0.7978845608028654f * (z0 + 0.044715f * z0 * z0 * z0);
            float4 a1 = 0.7978845608028654f * (z1 + 0.044715f * z1 * z1 * z1);
            float4 a2 = 0.7978845608028654f * (z2 + 0.044715f * z2 * z2 * z2);
            float4 a3 = 0.7978845608028654f * (z3 + 0.044715f * z3 * z3 * z3);
            float4 y0 = z0 * (1.0f / (1.0f + exp2(-2.8853900817779268f * a0)));
            float4 y1 = z1 * (1.0f / (1.0f + exp2(-2.8853900817779268f * a1)));
            float4 y2 = z2 * (1.0f / (1.0f + exp2(-2.8853900817779268f * a2)));
            float4 y3 = z3 * (1.0f / (1.0f + exp2(-2.8853900817779268f * a3)));
            ((device half4*)out)[base4] = half4(y0);
            ((device half4*)out)[base4 + 1u] = half4(y1);
            ((device half4*)out)[base4 + 2u] = half4(y2);
            ((device half4*)out)[base4 + 3u] = half4(y3);
        }
    } else {
        uint rows = min(4u, B - row0);
        uint cols = min(16u, D - col);
        for (uint r = 0u; r < rows; ++r) {
            uint base = (row0 + r) * D + col;
            for (uint i = 0u; i < cols; ++i) {
                float z = float(x[base + i]) + float(parameter[col + i]);
                float a = 0.7978845608028654f * (z + 0.044715f * z * z * z);
                float y = z * (1.0f / (1.0f + exp2(-2.8853900817779268f * a)));
                out[base + i] = half(y);
            }
        }
    }
}
