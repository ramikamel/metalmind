#include <metal_stdlib>
using namespace metal;
static inline float gelu_tanh_scalar(float z) {
    float z2 = z * z;
    float u = 0.7978845608028654f * z * (1.0f + 0.044715f * z2);
    return 0.5f * z * (1.0f + tanh(u));
}
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 gid [[thread_position_in_grid]]) {
    (void)epsilon;
    uint row = gid.y;
    uint col4 = gid.x;
    if (row >= B || col4 * 4u >= D) return;
    uint d4 = D >> 2u;
    uint idx4 = row * d4 + col4;
    device const float4* x4 = reinterpret_cast<device const float4*>(x);
    device const float4* p4 = reinterpret_cast<device const float4*>(parameter);
    device float4* o4 = reinterpret_cast<device float4*>(out);
    float4 z = x4[idx4] + p4[col4];
    float4 y;
    y.x = gelu_tanh_scalar(z.x);
    y.y = gelu_tanh_scalar(z.y);
    y.z = gelu_tanh_scalar(z.z);
    y.w = gelu_tanh_scalar(z.w);
    o4[idx4] = y;
}
