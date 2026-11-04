#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 gid [[thread_position_in_grid]]) {
    (void)epsilon;
    if (gid.y >= B || gid.x >= (D + 3u) / 4u) return;
    const float c = 0.7978845608028654f;
    if ((D & 3u) == 0u) {
        uint d4 = D >> 2u;
        uint idx4 = gid.y * d4 + gid.x;
        float4 xv = ((device const float4*)x)[idx4];
        float4 pv = ((device const float4*)parameter)[gid.x];
        float4 z = xv + pv;
        float4 r;
        float z0 = z.x; r.x = 0.5f * z0 * (1.0f + tanh(c * (z0 + 0.044715f * z0 * z0 * z0)));
        float z1 = z.y; r.y = 0.5f * z1 * (1.0f + tanh(c * (z1 + 0.044715f * z1 * z1 * z1)));
        float z2 = z.z; r.z = 0.5f * z2 * (1.0f + tanh(c * (z2 + 0.044715f * z2 * z2 * z2)));
        float z3 = z.w; r.w = 0.5f * z3 * (1.0f + tanh(c * (z3 + 0.044715f * z3 * z3 * z3)));
        ((device float4*)out)[idx4] = r;
    } else {
        uint row = gid.y;
        uint col0 = gid.x * 4u;
        for (uint i = 0u; i < 4u; ++i) {
            uint col = col0 + i;
            if (col < D) {
                uint idx = row * D + col;
                float z = x[idx] + parameter[col];
                out[idx] = 0.5f * z * (1.0f + tanh(c * (z + 0.044715f * z * z * z)));
            }
        }
    }
}
