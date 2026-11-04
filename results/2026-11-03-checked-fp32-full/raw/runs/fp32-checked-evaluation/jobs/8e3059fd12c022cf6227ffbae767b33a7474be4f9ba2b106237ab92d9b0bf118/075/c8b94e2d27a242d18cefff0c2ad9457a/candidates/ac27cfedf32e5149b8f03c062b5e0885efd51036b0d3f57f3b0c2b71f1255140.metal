#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint2 gid [[thread_position_in_grid]], uint2 grid [[threads_per_grid]]) {
    (void)epsilon;
    if (B == 0u || D == 0u) {
        return;
    }
    const float coeff = 0.7978845608028654f;
    if ((D & 3u) == 0u) {
        uint vec_per_row = D >> 2;
        device const float4* x4 = reinterpret_cast<device const float4*>(x);
        device const float4* p4 = reinterpret_cast<device const float4*>(parameter);
        device float4* o4 = reinterpret_cast<device float4*>(out);
        for (uint cv = gid.x; cv < vec_per_row; cv += grid.x) {
            float4 p = p4[cv];
            for (uint r = gid.y; r < B; r += grid.y) {
                uint i = r * vec_per_row + cv;
                float4 z = x4[i] + p;
                float4 z2 = z * z;
                float4 inner = coeff * (z + 0.044715f * z * z2);
                float4 y = 0.5f * z * (1.0f + tanh(inner));
                o4[i] = y;
            }
        }
    } else {
        for (uint c = gid.x; c < D; c += grid.x) {
            float p = parameter[c];
            for (uint r = gid.y; r < B; r += grid.y) {
                uint i = r * D + c;
                float z = x[i] + p;
                float z2 = z * z;
                float inner = coeff * (z + 0.044715f * z * z2);
                float y = 0.5f * z * (1.0f + tanh(inner));
                out[i] = y;
            }
        }
    }
}
