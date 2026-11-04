#include <metal_stdlib>
using namespace metal;
inline float gelu_tanh(float z) {
    float z2 = z * z;
    float inner = 0.7978845608028654f * (z + 0.044715f * z2 * z);
    return 0.5f * z * (1.0f + tanh(inner));
}
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 gid [[thread_position_in_grid]], uint3 grid [[threads_per_grid]]) {
    (void)epsilon;
    if ((D & 3u) == 0u) {
        device const float4* x4 = reinterpret_cast<device const float4*>(x);
        device const float4* p4 = reinterpret_cast<device const float4*>(parameter);
        device float4* o4 = reinterpret_cast<device float4*>(out);
        uint d4 = D >> 2;
        for (uint row = gid.y; row < B; row += grid.y) {
            uint base = row * d4;
            for (uint c = gid.x; c < d4; c += grid.x) {
                float4 z = x4[base + c] + p4[c];
                float4 r;
                r.x = gelu_tanh(z.x);
                r.y = gelu_tanh(z.y);
                r.z = gelu_tanh(z.z);
                r.w = gelu_tanh(z.w);
                o4[base + c] = r;
            }
        }
    } else {
        for (uint row = gid.y; row < B; row += grid.y) {
            uint base = row * D;
            for (uint c = gid.x; c < D; c += grid.x) {
                float z = x[base + c] + parameter[c];
                out[base + c] = gelu_tanh(z);
            }
        }
    }
}
