#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint tid [[thread_position_in_grid]]) {
    (void)epsilon;
    uint n = B * D;
    uint base = tid * 16u;
    if (base >= n) return;
    if (((D & 15u) == 0u) && (base + 16u <= n)) {
        uint row = base / D;
        uint col = base - row * D;
        uint vecBase = base >> 2;
        uint vecCol = col >> 2;
        device const float4* x4 = reinterpret_cast<device const float4*>(x);
        device const float4* p4 = reinterpret_cast<device const float4*>(parameter);
        device float4* o4 = reinterpret_cast<device float4*>(out);
        for (uint k = 0u; k < 4u; ++k) {
            float4 z = x4[vecBase + k] + p4[vecCol + k];
            float4 z3 = z * z * z;
            float4 arg = 0.7978845608028654f * (z + 0.044715f * z3);
            float4 y = 0.5f * z * (1.0f + tanh(arg));
            o4[vecBase + k] = y;
        }
        return;
    }
    for (uint k = 0u; k < 16u; ++k) {
        uint idx = base + k;
        if (idx >= n) return;
        uint col = idx - (idx / D) * D;
        float z = x[idx] + parameter[col];
        float z3 = z * z * z;
        float arg = 0.7978845608028654f * (z + 0.044715f * z3);
        out[idx] = 0.5f * z * (1.0f + tanh(arg));
    }
}
