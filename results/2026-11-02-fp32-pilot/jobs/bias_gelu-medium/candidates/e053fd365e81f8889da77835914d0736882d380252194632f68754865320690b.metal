#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint2 gid [[thread_position_in_grid]], uint2 grid_size [[threads_per_grid]]) {
    (void)epsilon;
    if ((D & 3u) == 0u) {
        uint D4 = D >> 2u;
        device const float4* x4 = reinterpret_cast<device const float4*>(x);
        device const float4* p4 = reinterpret_cast<device const float4*>(parameter);
        device float4* o4 = reinterpret_cast<device float4*>(out);
        for (uint row = gid.y; row < B; row += grid_size.y) {
            uint base = row * D4;
            for (uint col = gid.x; col < D4; col += grid_size.x) {
                float4 z = x4[base + col] + p4[col];
                float4 z2 = z * z;
                float4 inner = 0.7978845608f * (z + 0.044715f * z2 * z);
                o4[base + col] = 0.5f * z * (1.0f + tanh(inner));
            }
        }
    } else {
        uint total = B * D;
        uint flat = gid.y * grid_size.x + gid.x;
        uint stride = grid_size.x * grid_size.y;
        for (uint idx = flat; idx < total; idx += stride) {
            uint col = (D == 0u) ? 0u : (idx % D);
            float z = x[idx] + parameter[col];
            float inner = 0.7978845608f * (z + 0.044715f * z * z * z);
            out[idx] = 0.5f * z * (1.0f + tanh(inner));
        }
    }
}
