#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint2 gid [[thread_position_in_grid]]) {
    (void)epsilon;
    uint col = gid.x * 4u;
    uint row = gid.y;
    if (col >= D || row >= B) return;
    uint d4 = D >> 2;
    uint idx = row * d4 + gid.x;
    float4 xv = ((device const float4*)x)[idx];
    float4 pv = ((device const float4*)parameter)[gid.x];
    float4 z = xv + pv;
    float4 z3 = z * z * z;
    float4 a = 0.7978845608028654f * (z + 0.044715f * z3);
    float4 th = float4(tanh(a.x), tanh(a.y), tanh(a.z), tanh(a.w));
    float4 y = 0.5f * z * (1.0f + th);
    ((device float4*)out)[idx] = y;
}
