#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint2 gid [[thread_position_in_grid]]) {
    (void)epsilon;
    if (gid.x >= D || gid.y >= 16u) {
        return;
    }
    float bias = parameter[gid.x];
    for (uint row = gid.y; row < B; row += 16u) {
        uint idx = row * D + gid.x;
        float z = x[idx] + bias;
        float cube = z * z * z;
        float inner = 0.7978845608028654f * (z + 0.044715f * cube);
        out[idx] = 0.5f * z * (1.0f + tanh(inner));
    }
}
