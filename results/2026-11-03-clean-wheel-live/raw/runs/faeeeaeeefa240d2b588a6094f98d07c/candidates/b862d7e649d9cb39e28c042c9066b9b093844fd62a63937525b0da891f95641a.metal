#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint2 gid [[thread_position_in_grid]]) {
    (void)epsilon;
    if (gid.y >= B || gid.x >= (D + 3u) / 4u) {
        return;
    }
    uint col = gid.x * 4u;
    uint row_base = gid.y * D;
    if (col + 3u < D) {
        device const float4* x4 = reinterpret_cast<device const float4*>(x);
        device const float4* p4 = reinterpret_cast<device const float4*>(parameter);
        device float4* o4 = reinterpret_cast<device float4*>(out);
        uint vec_index = gid.y * (D / 4u) + gid.x;
        float4 z = x4[vec_index] + p4[gid.x];
        float4 z2 = z * z;
        float4 inner = 0.7978845608028654f * (z + 0.044715f * z2 * z);
        float4 y = 0.5f * z * (1.0f + tanh(inner));
        o4[vec_index] = y;
    } else {
        for (uint c = col; c < D; ++c) {
            float z = x[row_base + c] + parameter[c];
            float z2 = z * z;
            float inner = 0.7978845608028654f * (z + 0.044715f * z2 * z);
            out[row_base + c] = 0.5f * z * (1.0f + tanh(inner));
        }
    }
}
