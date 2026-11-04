#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device const half* parameter, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint2 gid [[thread_position_in_grid]]) {
    (void)epsilon;
    uint col = gid.x;
    uint row0 = gid.y * 4u;
    if (col >= D || row0 >= B) return;
    float bias = float(parameter[col]);
    uint base = row0 * D + col;
    for (uint i = 0u; i < 4u; ++i) {
        uint row = row0 + i;
        if (row >= B) break;
        float z = float(x[base]) + bias;
        float z2 = z * z;
        float u = 0.7978845608028654f * z * (1.0f + 0.044715f * z2);
        float y = 0.5f * z * (1.0f + tanh(u));
        out[base] = half(y);
        base += D;
    }
}
