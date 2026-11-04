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
        for (uint row = gid.y; row < B; row += tpg.y) {
            const uint base = row * d4;
            for (uint c = gid.x; c < d4; c += tpg.x) {
                float4 z = float4(x4[base + c]) + float4(p4[c]);
                float4 z3 = z * z * z;
                float4 y = 0.5f * z * (1.0f + tanh(k * (z + 0.044715f * z3)));
                o4[base + c] = half4(y);
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
