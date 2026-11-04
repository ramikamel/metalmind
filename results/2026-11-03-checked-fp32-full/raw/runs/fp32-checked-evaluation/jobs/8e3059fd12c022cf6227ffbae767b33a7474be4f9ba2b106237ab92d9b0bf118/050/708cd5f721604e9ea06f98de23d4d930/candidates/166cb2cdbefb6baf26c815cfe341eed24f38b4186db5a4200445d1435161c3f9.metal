#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint2 gid [[thread_position_in_grid]], uint2 grid [[threads_per_grid]]) {
    (void)epsilon;
    if ((D & 3u) != 0u) {
        uint total = B * D;
        uint stride = grid.x * grid.y;
        uint start = gid.y * grid.x + gid.x;
        for (uint i = start; i < total; i += stride) {
            uint c = i % D;
            float z = x[i] + parameter[c];
            float inner = 0.7978845608028654f * (z + 0.044715f * z * z * z);
            out[i] = 0.5f * z * (1.0f + tanh(inner));
        }
        return;
    }
    uint vecD = D >> 2;
    device const float4* x4 = reinterpret_cast<device const float4*>(x);
    device float4* out4 = reinterpret_cast<device float4*>(out);
    device const float4* p4 = reinterpret_cast<device const float4*>(parameter);
    for (uint col = gid.x; col < vecD; col += grid.x) {
        float4 bias = p4[col];
        for (uint rowBase = gid.y << 3; rowBase < B; rowBase += grid.y << 3) {
            for (uint r = 0u; r < 8u; ++r) {
                uint row = rowBase + r;
                if (row >= B) break;
                uint idx = row * vecD + col;
                float4 z = x4[idx] + bias;
                float4 z2 = z * z;
                float4 inner = 0.7978845608028654f * (z + 0.044715f * z2 * z);
                out4[idx] = 0.5f * z * (1.0f + tanh(inner));
            }
        }
    }
}
