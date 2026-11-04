#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint2 gid [[thread_position_in_grid]], uint2 grid [[threads_per_grid]]) {
    (void)epsilon;
    if ((D & 3u) == 0u) {
        uint row = gid.y;
        uint col4 = gid.x;
        uint blocks = D >> 2;
        if (row >= B || col4 >= blocks) return;
        uint idx4 = row * blocks + col4;
        float4 z = ((device const float4*)x)[idx4] + ((device const float4*)parameter)[col4];
        float4 c = z + 0.044715f * z * z * z;
        float4 inner = 0.7978845608028654f * c;
        float4 y;
        y.x = 0.5f * z.x * (1.0f + tanh(inner.x));
        y.y = 0.5f * z.y * (1.0f + tanh(inner.y));
        y.z = 0.5f * z.z * (1.0f + tanh(inner.z));
        y.w = 0.5f * z.w * (1.0f + tanh(inner.w));
        ((device float4*)out)[idx4] = y;
    } else {
        uint tid = gid.y * grid.x + gid.x;
        uint base = tid * 4u;
        uint total = B * D;
        for (uint i = 0u; i < 4u; ++i) {
            uint idx = base + i;
            if (idx >= total) return;
            uint col = idx % D;
            float z = x[idx] + parameter[col];
            float inner = 0.7978845608028654f * (z + 0.044715f * z * z * z);
            out[idx] = 0.5f * z * (1.0f + tanh(inner));
        }
    }
}
