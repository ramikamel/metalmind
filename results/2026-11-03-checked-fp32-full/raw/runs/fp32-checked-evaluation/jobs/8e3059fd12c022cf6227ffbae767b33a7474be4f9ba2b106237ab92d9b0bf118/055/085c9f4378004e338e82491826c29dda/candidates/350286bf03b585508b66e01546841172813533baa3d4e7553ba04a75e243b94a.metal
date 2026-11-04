#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 gid [[thread_position_in_grid]], uint3 gridDim [[threads_per_grid]]) {
    (void)epsilon;
    uint tid = gid.x;
    uint stride = gridDim.x;
    uint total = B * D;
    if ((D & 3u) == 0u) {
        uint vecTotal = total >> 2;
        uint D4 = D >> 2;
        device const float4* x4 = (device const float4*)x;
        device const float4* p4 = (device const float4*)parameter;
        device float4* o4 = (device float4*)out;
        for (uint i = tid; i < vecTotal; i += stride) {
            float4 z = x4[i] + p4[i % D4];
            float4 inner = 0.7978845608028654f * (z + 0.044715f * z * z * z);
            float4 t = float4(tanh(inner.x), tanh(inner.y), tanh(inner.z), tanh(inner.w));
            o4[i] = 0.5f * z * (1.0f + t);
        }
    } else {
        for (uint i = tid; i < total; i += stride) {
            float z = x[i] + parameter[i % D];
            float inner = 0.7978845608028654f * (z + 0.044715f * z * z * z);
            out[i] = 0.5f * z * (1.0f + tanh(inner));
        }
    }
}
