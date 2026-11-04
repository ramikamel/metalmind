#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint2 gid [[thread_position_in_grid]]) {
    (void)epsilon;
    if (gid.x >= D) return;
    uint row_base = gid.y * 4u;
    if (row_base >= B) return;
    uint col = gid.x;
    float p = parameter[col];
    for (uint i = 0u; i < 4u; ++i) {
        uint row = row_base + i;
        if (row >= B) return;
        uint idx = row * D + col;
        float z = x[idx] + p;
        float z2 = z * z;
        float u = 0.7978845608028654f * z * (1.0f + 0.044715f * z2);
        float uc = clamp(u, -10.0f, 10.0f);
        float e = exp2(-2.885390081777927f * uc);
        out[idx] = z / (1.0f + e);
    }
}
