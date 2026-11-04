#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint gid [[thread_position_in_grid]], uint tpg [[threads_per_grid]]) {
    (void)epsilon;
    uint total = B * D;
    if ((D & 7u) == 0u) {
        uint cols4 = D >> 2u;
        uint total4 = total >> 2u;
        device const float4* x4 = reinterpret_cast<device const float4*>(x);
        device const float4* p4 = reinterpret_cast<device const float4*>(parameter);
        device float4* o4 = reinterpret_cast<device float4*>(out);
        for (uint base = gid << 1u; base < total4; base += tpg << 1u) {
            uint pc = base % cols4;
            float4 pz0 = p4[pc];
            float4 pz1 = p4[pc + 1u];
            float4 z0 = x4[base] + pz0;
            float4 z1 = x4[base + 1u] + pz1;
            float4 a0 = -2.30258509f * z0 * (1.0f + 0.044715f * z0 * z0);
            float4 a1 = -2.30258509f * z1 * (1.0f + 0.044715f * z1 * z1);
            float4 s0 = 1.0f / (1.0f + exp2(a0));
            float4 s1 = 1.0f / (1.0f + exp2(a1));
            o4[base] = z0 * s0;
            o4[base + 1u] = z1 * s1;
        }
    } else {
        for (uint i = gid; i < total; i += tpg) {
            uint c = (D == 0u) ? 0u : (i % D);
            float z = x[i] + parameter[c];
            float inner = 0.7978845608028654f * (z + 0.044715f * z * z * z);
            out[i] = 0.5f * z * (1.0f + tanh(inner));
        }
    }
}
