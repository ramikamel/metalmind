#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint gid [[thread_position_in_grid]]) {
    (void)epsilon;
    uint total = B * D;
    uint base = gid * 4u;
    if (base >= total || D == 0u) {
        return;
    }
    uint row = base / D;
    uint col = base - row * D;
    if (col + 3u < D && (col & 3u) == 0u) {
        float4 xv = ((device const float4*)x)[gid];
        float4 pv = ((device const float4*)parameter)[col / 4u];
        float4 z = xv + pv;
        float4 inner = 0.7978845608028654f * (z + 0.044715f * z * z * z);
        float4 y = 0.5f * z * (1.0f + tanh(inner));
        ((device float4*)out)[gid] = (float4)y;
    } else {
        for (uint i = 0u; i < 4u; ++i) {
            uint idx = base + i;
            if (idx < total) {
                uint r = idx / D;
                uint c = idx - r * D;
                float z = float(x[idx]) + float(parameter[c]);
                float inner = 0.7978845608028654f * (z + 0.044715f * z * z * z);
                float y = 0.5f * z * (1.0f + tanh(inner));
                out[idx] = float(y);
            }
        }
    }
}
