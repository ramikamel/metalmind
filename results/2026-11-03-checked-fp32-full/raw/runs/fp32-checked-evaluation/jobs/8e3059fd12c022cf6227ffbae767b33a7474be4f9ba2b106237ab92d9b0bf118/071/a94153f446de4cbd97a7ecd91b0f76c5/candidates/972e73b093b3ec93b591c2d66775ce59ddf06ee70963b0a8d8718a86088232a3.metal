#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint2 gid [[thread_position_in_grid]]) {
    (void)epsilon;
    uint vecD = D >> 2;
    if (gid.y >= B || gid.x >= vecD) {
        return;
    }
    device const float4* x4 = reinterpret_cast<device const float4*>(x);
    device const float4* p4 = reinterpret_cast<device const float4*>(parameter);
    device float4* o4 = reinterpret_cast<device float4*>(out);
    uint idx = gid.y * vecD + gid.x;
    float4 z = x4[idx] + p4[gid.x];
    float4 z2 = z * z;
    float4 inner = 0.7978845608028654f * (z + 0.044715f * z2 * z);
    float4 y = 0.5f * z * (1.0f + tanh(inner));
    o4[idx] = y;
}
