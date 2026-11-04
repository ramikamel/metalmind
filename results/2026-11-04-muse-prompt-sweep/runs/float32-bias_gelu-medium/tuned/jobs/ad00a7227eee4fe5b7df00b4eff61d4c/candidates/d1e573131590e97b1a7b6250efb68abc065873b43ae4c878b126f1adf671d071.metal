#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint2 gid [[thread_position_in_grid]], uint2 grid [[threads_per_grid]]) {
    (void)epsilon;
    if ((D & 3u) == 0u && D != 0u) {
        uint blocks = D >> 2;
        device const float4* x4 = (device const float4*)x;
        device const float4* p4 = (device const float4*)parameter;
        device float4* o4 = (device float4*)out;
        for (uint row = gid.y; row < B; row += grid.y) {
            uint rowBase = row * blocks;
            for (uint col4 = gid.x; col4 < blocks; col4 += grid.x) {
                uint idx4 = rowBase + col4;
                float4 z = x4[idx4] + p4[col4];
                float4 poly = z + 0.044715f * z * z * z;
                float4 inner = 0.7978845608028654f * poly;
                float4 e = fast::exp2(2.885390081777927f * inner);
                float4 y = z * (1.0f - 1.0f / (e + 1.0f));
                o4[idx4] = y;
            }
        }
    } else if (D != 0u) {
        uint tid = gid.y * grid.x + gid.x;
        uint stride = grid.x * grid.y;
        uint total = B * D;
        for (uint idx = tid; idx < total; idx += stride) {
            uint col = idx % D;
            float z = x[idx] + parameter[col];
            float poly = z + 0.044715f * z * z * z;
            float inner = 0.7978845608028654f * poly;
            float e = fast::exp2(2.885390081777927f * inner);
            out[idx] = z * (1.0f - 1.0f / (e + 1.0f));
        }
    }
}
