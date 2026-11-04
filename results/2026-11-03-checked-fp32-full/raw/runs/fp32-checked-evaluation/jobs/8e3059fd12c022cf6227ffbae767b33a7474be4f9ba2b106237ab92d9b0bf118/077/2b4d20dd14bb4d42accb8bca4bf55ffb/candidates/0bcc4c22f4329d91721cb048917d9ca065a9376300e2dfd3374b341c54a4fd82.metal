#include <metal_stdlib>
using namespace metal;
inline float gelu_tanh(float z) {
    float z2 = z * z;
    float inner = 0.7978845608028654f * (z + 0.044715f * z2 * z);
    return 0.5f * z * (1.0f + tanh(inner));
}
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint tid [[thread_position_in_grid]], uint grid [[threads_per_grid]]) {
    (void)epsilon;
    uint total = B * D;
    if (total == 0u || D == 0u) {
        return;
    }
    if ((D & 3u) == 0u) {
        uint total_vec = total >> 2;
        device const float4* x4 = (device const float4*)x;
        device const float4* p4 = (device const float4*)parameter;
        device float4* o4 = (device float4*)out;
        for (uint v = tid; v < total_vec; v += grid) {
            uint col = (v << 2) % D;
            float4 z = x4[v] + p4[col >> 2];
            float4 r;
            r.x = gelu_tanh(z.x);
            r.y = gelu_tanh(z.y);
            r.z = gelu_tanh(z.z);
            r.w = gelu_tanh(z.w);
            o4[v] = r;
        }
        uint start = total_vec << 2;
        for (uint i = start + tid; i < total; i += grid) {
            uint col = i % D;
            out[i] = gelu_tanh(x[i] + parameter[col]);
        }
    } else {
        for (uint i = tid; i < total; i += grid) {
            uint col = i % D;
            out[i] = gelu_tanh(x[i] + parameter[col]);
        }
    }
}
