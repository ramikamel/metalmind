#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint2 gid [[thread_position_in_grid]]) {
    (void)epsilon;
    if (gid.x >= D || gid.y >= B) {
        return;
    }
    uint idx = gid.y * D + gid.x;
    float z = x[idx] + parameter[gid.x];
    float z2 = z * z;
    float inner = 0.7978845608028654f * (z + 0.044715f * z * z2);
    float y = 0.5f * z * (1.0f + tanh(inner));
    out[idx] = y;
}
