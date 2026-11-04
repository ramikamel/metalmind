#include <metal_stdlib>
using namespace metal;
inline float gelu_tanh(float z) {
    float z2 = z * z;
    float u = 0.7978845608028654f * (z + 0.044715f * z * z2);
    return 0.5f * z * (1.0f + tanh(u));
}
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint tid [[thread_position_in_grid]], uint tpg [[threads_per_grid]]) {
    (void)epsilon;
    uint total = B * D;
    if (((D & 3u) == 0u) && ((total & 3u) == 0u)) {
        uint d4 = D >> 2;
        uint total4 = B * d4;
        device const float4* x4 = (device const float4*)x;
        device const float4* p4 = (device const float4*)parameter;
        device float4* o4 = (device float4*)out;
        for (uint i = tid; i < total4; i += tpg) {
            uint row = i / d4;
            uint col = i - row * d4;
            float4 z = x4[i] + p4[col];
            float4 r;
            r.x = gelu_tanh(z.x);
            r.y = gelu_tanh(z.y);
            r.z = gelu_tanh(z.z);
            r.w = gelu_tanh(z.w);
            o4[i] = r;
        }
    } else {
        for (uint i = tid; i < total; i += tpg) {
            uint col = (D == 0u) ? 0u : (i % D);
            float z = x[i] + parameter[col];
            out[i] = gelu_tanh(z);
        }
    }
}