#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint2 gid [[thread_position_in_grid]]) {
    (void)epsilon;
    uint col = gid.x << 2;
    uint row0 = gid.y << 2;
    if (col >= D || row0 >= B) {
        return;
    }
    if (col + 4u <= D && (D & 3u) == 0u) {
        device const float4* x4 = reinterpret_cast<device const float4*>(x);
        device float4* out4 = reinterpret_cast<device float4*>(out);
        device const float4* p4 = reinterpret_cast<device const float4*>(parameter);
        float4 bias = p4[gid.x];
        uint vecStride = D >> 2;
        for (uint r = 0u; r < 4u; ++r) {
            uint row = row0 + r;
            if (row >= B) {
                break;
            }
            uint idx = row * vecStride + gid.x;
            float4 z = x4[idx] + bias;
            float4 z2 = z * z;
            float4 inner = 0.7978845608028654f * (z + 0.044715f * z2 * z);
            float4 y = 0.5f * z * (1.0f + tanh(inner));
            out4[idx] = y;
        }
    } else {
        for (uint r = 0u; r < 4u; ++r) {
            uint row = row0 + r;
            if (row >= B) {
                break;
            }
            uint base = row * D;
            for (uint c = col; c < D && c < col + 4u; ++c) {
                float z = x[base + c] + parameter[c];
                float inner = 0.7978845608028654f * (z + 0.044715f * z * z * z);
                out[base + c] = 0.5f * z * (1.0f + tanh(inner));
            }
        }
    }
}
