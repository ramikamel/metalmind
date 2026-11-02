#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint2 gid [[thread_position_in_grid]]) {
    (void)epsilon;
    uint col4 = gid.x;
    uint d4 = D >> 2;
    if (col4 >= d4) return;
    uint row0 = gid.y * 8u;
    if (row0 >= B) return;
    float4 pv = ((device const float4*)parameter)[col4];
    device const float4* x4 = (device const float4*)x;
    device float4* o4 = (device float4*)out;
    for (uint i = 0u; i < 8u; ++i) {
        uint row = row0 + i;
        if (row >= B) break;
        uint idx = row * d4 + col4;
        float4 z = x4[idx] + pv;
        float4 a = 0.7978845608028654f * (z + 0.044715f * (z * z * z));
        float4 th = fast::tanh(a);
        o4[idx] = 0.5f * z * (1.0f + th);
    }
}
