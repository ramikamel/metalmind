#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint gid [[thread_position_in_grid]], uint grid_size [[threads_per_grid]]) {
    (void)epsilon;
    uint total = B * D;
    if ((D & 3u) == 0u) {
        uint total4 = total >> 2u;
        uint D4 = D >> 2u;
        for (uint idx = gid; idx < total4; idx += grid_size) {
            uint row = idx / D4;
            uint col = idx - row * D4;
            float4 xv = reinterpret_cast<device const float4*>(x)[idx];
            float4 pv = reinterpret_cast<device const float4*>(parameter)[col];
            float4 z = xv + pv;
            float4 z2 = z * z;
            float4 inner = 0.7978845608f * (z + 0.044715f * z2 * z);
            float4 y = 0.5f * z * (1.0f + tanh(inner));
            reinterpret_cast<device float4*>(out)[idx] = y;
        }
    } else {
        for (uint idx = gid; idx < total; idx += grid_size) {
            uint col = (D == 0u) ? 0u : (idx % D);
            float z = x[idx] + parameter[col];
            float inner = 0.7978845608f * (z + 0.044715f * z * z * z);
            out[idx] = 0.5f * z * (1.0f + tanh(inner));
        }
    }
}
