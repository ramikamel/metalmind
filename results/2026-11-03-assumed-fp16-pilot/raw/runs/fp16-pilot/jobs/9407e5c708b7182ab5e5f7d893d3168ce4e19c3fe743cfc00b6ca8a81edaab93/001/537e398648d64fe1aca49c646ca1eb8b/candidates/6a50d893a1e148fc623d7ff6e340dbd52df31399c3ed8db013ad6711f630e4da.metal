#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device const half* parameter, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint2 gid [[thread_position_in_grid]], uint2 tpg [[threads_per_grid]]) {
    (void)epsilon;
    const float k = 0.7978845608028654f;
    if ((D & 3u) == 0u) {
        const uint d4 = D >> 2;
        device const half4* x4 = (device const half4*)x;
        device const half4* p4 = (device const half4*)parameter;
        device half4* o4 = (device half4*)out;
        const uint step = tpg.x << 1;
        for (uint row = gid.y; row < B; row += tpg.y) {
            const uint base = row * d4;
            for (uint c = gid.x; c < d4; c += step) {
                float4 z0 = float4(x4[base + c]) + float4(p4[c]);
                float4 z03 = z0 * z0 * z0;
                float4 y0 = 0.5f * z0 * (1.0f + tanh(k * (z0 + 0.044715f * z03)));
                o4[base + c] = half4(y0);
                const uint c1 = c + tpg.x;
                if (c1 < d4) {
                    float4 z1 = float4(x4[base + c1]) + float4(p4[c1]);
                    float4 z13 = z1 * z1 * z1;
                    float4 y1 = 0.5f * z1 * (1.0f + tanh(k * (z1 + 0.044715f * z13)));
                    o4[base + c1] = half4(y1);
                }
            }
        }
    } else {
        for (uint row = gid.y; row < B; row += tpg.y) {
            const uint base = row * D;
            for (uint col = gid.x; col < D; col += tpg.x) {
                float z = float(x[base + col]) + float(parameter[col]);
                float z3 = z * z * z;
                float y = 0.5f * z * (1.0f + tanh(k * (z + 0.044715f * z3)));
                out[base + col] = half(y);
            }
        }
    }
}
