#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint2 gid [[thread_position_in_grid]]) {
    (void)epsilon;
    if (gid.x >= D) return;
    uint row0 = gid.y * 4u;
    if (row0 >= B) return;
    float p = parameter[gid.x];
    uint rows = min(4u, B - row0);
    for (uint i = 0u; i < rows; ++i) {
        uint idx = (row0 + i) * D + gid.x;
        float z = x[idx] + p;
        float z2 = z * z;
        float inner = 0.7978845608028654f * (z + 0.044715f * z * z2);
        out[idx] = 0.5f * z * (1.0f + tanh(inner));
    }
}
