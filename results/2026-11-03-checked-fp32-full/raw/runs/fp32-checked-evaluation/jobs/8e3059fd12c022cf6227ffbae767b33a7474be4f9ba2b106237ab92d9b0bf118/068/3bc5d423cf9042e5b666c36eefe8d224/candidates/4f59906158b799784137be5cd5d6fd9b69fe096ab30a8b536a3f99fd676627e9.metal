#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint2 gid [[thread_position_in_grid]], uint2 gsize [[threads_per_grid]]) {
    (void)epsilon;
    const float kSqrt2OverPi = 0.7978845608028654f;
    const float kCoeff = 0.044715f;
    if ((D & 3u) == 0u) {
        const uint blocksPerRow = D >> 2;
        device const float4* x4 = reinterpret_cast<device const float4*>(x);
        device const float4* p4 = reinterpret_cast<device const float4*>(parameter);
        device float4* o4 = reinterpret_cast<device float4*>(out);
        for (uint row = gid.y; row < B; row += gsize.y) {
            const uint rowBase = row * blocksPerRow;
            for (uint block = gid.x; block < blocksPerRow; block += gsize.x) {
                float4 z = x4[rowBase + block] + p4[block];
                float4 z3 = z * z * z;
                float4 inner = kSqrt2OverPi * (z + kCoeff * z3);
                float4 y = 0.5f * z * (1.0f + tanh(inner));
                o4[rowBase + block] = y;
            }
        }
    } else {
        const uint totalThreads = gsize.x * gsize.y;
        const uint linearTid = gid.y * gsize.x + gid.x;
        const uint total = B * D;
        for (uint idx = linearTid; idx < total; idx += totalThreads) {
            uint col = (D == 0u) ? 0u : (idx % D);
            float z = x[idx] + parameter[col];
            float z3 = z * z * z;
            float inner = kSqrt2OverPi * (z + kCoeff * z3);
            float y = 0.5f * z * (1.0f + tanh(inner));
            out[idx] = y;
        }
    }
}
