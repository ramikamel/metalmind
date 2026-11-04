#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device const half* parameter, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint2 gid [[thread_position_in_grid]]) {
    (void)epsilon;
    uint col = gid.x * 4u;
    uint row = gid.y;
    if (row >= B || col >= D) return;
    uint base = row * D + col;
    uint n = D - col;
    if (n > 4u) n = 4u;
    for (uint i = 0u; i < n; ++i) {
        float z = float(x[base + i]) + float(parameter[col + i]);
        float z2 = z * z;
        float u = 0.7978845608028654f * z * (1.0f + 0.044715f * z2);
        float y = 0.5f * z * (1.0f + tanh(u));
        out[base + i] = half(y);
    }
}
