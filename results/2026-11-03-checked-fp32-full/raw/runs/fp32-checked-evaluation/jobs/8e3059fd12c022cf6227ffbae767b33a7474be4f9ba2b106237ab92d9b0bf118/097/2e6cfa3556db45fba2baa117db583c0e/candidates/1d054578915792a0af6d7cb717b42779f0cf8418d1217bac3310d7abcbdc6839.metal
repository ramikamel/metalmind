#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint gid [[thread_position_in_grid]], uint gridDim [[threads_per_grid]]) {
    (void)epsilon;
    uint total = B * D;
    const float kSqrt2OverPi = 0.7978845608028654f;
    if ((D & 3u) == 0u && total >= 4u) {
        uint numVec = total >> 2u;
        uint colsVec = D >> 2u;
        device const float4* x4 = reinterpret_cast<device const float4*>(x);
        device const float4* p4 = reinterpret_cast<device const float4*>(parameter);
        device float4* o4 = reinterpret_cast<device float4*>(out);
        for (uint idx = gid; idx < numVec; idx += gridDim) {
            uint colVec = idx % colsVec;
            float4 z = x4[idx] + p4[colVec];
            float4 z2 = z * z;
            float4 inner = kSqrt2OverPi * (z + 0.044715f * z2 * z);
            float4 y = 0.5f * z * (1.0f + tanh(inner));
            o4[idx] = y;
        }
        uint tailStart = numVec << 2u;
        if (gid == 0u) {
            for (uint i = tailStart; i < total; ++i) {
                uint col = i % D;
                float z = x[i] + parameter[col];
                float inner = kSqrt2OverPi * (z + 0.044715f * z * z * z);
                out[i] = 0.5f * z * (1.0f + tanh(inner));
            }
        }
    } else {
        for (uint i = gid; i < total; i += gridDim) {
            uint col = (D == 0u) ? 0u : (i % D);
            float z = x[i] + parameter[col];
            float inner = kSqrt2OverPi * (z + 0.044715f * z * z * z);
            out[i] = 0.5f * z * (1.0f + tanh(inner));
        }
    }
}
