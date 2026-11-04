#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 position_in_grid [[thread_position_in_grid]], uint3 grid_size [[threads_per_grid]]) {
    uint tid = position_in_grid.x;
    uint grid = grid_size.x;
    const float k = 0.7978845608028654f;
    if (B == 256u && D == 1024u) {
        if (tid >= 16384u) return;
        uint col_block = tid & 255u;
        uint row_block = tid >> 8;
        uint col = col_block * 4u;
        device const float4* x4 = reinterpret_cast<device const float4*>(x);
        device const float4* p4 = reinterpret_cast<device const float4*>(parameter);
        device float4* o4 = reinterpret_cast<device float4*>(out);
        float4 p = p4[col_block];
        uint base = row_block * 1024u + col_block;
        for (uint r = 0u; r < 4u; ++r) {
            uint idx = base + r * 256u;
            float4 z = x4[idx] + p;
            float4 z3 = z * z * z;
            float4 y = 0.5f * z * (1.0f + tanh(k * (z + 0.044715f * z3)));
            o4[idx] = y;
        }
        return;
    }
    uint total = B * D;
    if (D == 0u) return;
    for (uint i = tid; i < total; i += grid) {
        uint col = i % D;
        float z = x[i] + parameter[col];
        float z3 = z * z * z;
        float y = 0.5f * z * (1.0f + tanh(k * (z + 0.044715f * z3)));
        out[i] = y;
    }
}
