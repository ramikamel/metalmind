#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint2 gid [[thread_position_in_grid]]) {
    (void)epsilon;
    uint col = gid.x * 4u;
    if (col >= D || gid.y >= B) return;
    uint idx = gid.y * D + col;
    float4 xv = reinterpret_cast<device const float4*>(x + idx)[0];
    float4 pv = reinterpret_cast<device const float4*>(parameter + col)[0];
    float4 z = xv + pv;
    float4 z2 = z * z;
    float4 inner = 0.7978845608028654f * (z + 0.044715f * z * z2);
    float4 y = 0.5f * z * (1.0f + tanh(inner));
    reinterpret_cast<device float4*>(out + idx)[0] = y;
}
