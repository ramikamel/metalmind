#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint gid [[thread_position_in_grid]]) {
    (void)epsilon;
    const uint totalThreads = 65536u;
    if ((D & 3u) == 0u) {
        const uint d4 = D >> 2u;
        const uint n4 = B * d4;
        device const float4* x4 = (device const float4*)x;
        device const float4* p4 = (device const float4*)parameter;
        device float4* o4 = (device float4*)out;
        for (uint i = gid; i < n4; i += totalThreads) {
            const uint row = i / d4;
            const uint col = i - row * d4;
            const float4 z = x4[i] + p4[col];
            const float4 z2 = z * z;
            const float4 t = z * (1.0f + 0.044715f * z2);
            const float4 e = fast::exp2(-2.30258509f * t);
            o4[i] = z / (1.0f + e);
        }
    } else {
        const uint n = B * D;
        for (uint i = gid; i < n; i += totalThreads) {
            const uint row = (D == 0u) ? 0u : i / D;
            const uint col = i - row * D;
            const float z = x[i] + parameter[col];
            const float t = z * (1.0f + 0.044715f * z * z);
            const float e = fast::exp2(-2.30258509f * t);
            out[i] = z / (1.0f + e);
        }
    }
}
