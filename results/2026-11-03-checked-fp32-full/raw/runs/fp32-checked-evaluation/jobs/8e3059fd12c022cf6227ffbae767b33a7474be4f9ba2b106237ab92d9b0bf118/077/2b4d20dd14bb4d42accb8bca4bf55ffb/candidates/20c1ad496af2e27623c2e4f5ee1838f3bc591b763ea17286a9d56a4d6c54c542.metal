#include <metal_stdlib>
using namespace metal;
static inline float gelu_tanh(float z) {
    float z2 = z * z;
    float inner = 0.7978845608028654f * (z + 0.044715f * z2 * z);
    return 0.5f * z * (1.0f + tanh(inner));
}
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 gid [[thread_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 tgid [[threadgroup_position_in_grid]]) {
    (void)epsilon;
    (void)tid;
    (void)tgid;
    uint total4 = (B * D) >> 2;
    uint base = gid.x << 2;
    device const float4* x4 = (device const float4*)x;
    device const float4* p4 = (device const float4*)parameter;
    device float4* o4 = (device float4*)out;
    for (uint i = 0; i < 4; ++i) {
        uint vi = base + i;
        if (vi < total4) {
            uint col4 = ((vi << 2) % D) >> 2;
            float4 z = x4[vi] + p4[col4];
            float4 y;
            y.x = gelu_tanh(z.x);
            y.y = gelu_tanh(z.y);
            y.z = gelu_tanh(z.z);
            y.w = gelu_tanh(z.w);
            o4[vi] = y;
        }
    }
}
