#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 gid [[thread_position_in_grid]], uint3 gdim [[threads_per_grid]]) {
    (void)epsilon;
    const float k = 0.7978845608028654f;
    if ((D & 3u) == 0u) {
        const uint vpr = D >> 2;
        device const float4* x4 = (device const float4*)x;
        device const float4* p4 = (device const float4*)parameter;
        device float4* o4 = (device float4*)out;
        for (uint cv = gid.x; cv < vpr; cv += gdim.x) {
            float4 b = p4[cv];
            for (uint r = gid.y; r < B; r += gdim.y) {
                uint idx = r * vpr + cv;
                float4 z = x4[idx] + b;
                float4 z2 = z * z;
                float4 inner = k * (z + 0.044715f * z2 * z);
                float4 y = 0.5f * z * (1.0f + tanh(inner));
                o4[idx] = y;
            }
        }
    } else {
        for (uint c = gid.x; c < D; c += gdim.x) {
            float b = parameter[c];
            for (uint r = gid.y; r < B; r += gdim.y) {
                uint idx = r * D + c;
                float z = x[idx] + b;
                float inner = k * (z + 0.044715f * z * z * z);
                out[idx] = 0.5f * z * (1.0f + tanh(inner));
            }
        }
    }
}
