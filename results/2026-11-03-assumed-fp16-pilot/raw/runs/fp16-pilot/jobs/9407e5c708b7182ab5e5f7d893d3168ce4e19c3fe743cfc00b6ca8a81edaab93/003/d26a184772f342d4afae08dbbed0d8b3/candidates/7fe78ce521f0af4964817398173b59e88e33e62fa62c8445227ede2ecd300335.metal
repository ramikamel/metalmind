#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device const half* parameter, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 gid [[thread_position_in_grid]], uint3 tpg [[threads_per_grid]]) {
    uint col = gid.x;
    if (col >= D) return;
    float bias = float(parameter[col]);
    (void)epsilon;
    for (uint row = gid.y; row < B; row += tpg.y) {
        uint idx = row * D + col;
        float z = float(x[idx]) + bias;
        float z2 = z * z;
        float inner = 0.7978845608028654f * fma(0.044715f, z2 * z, z);
        float y = 0.5f * z * (1.0f + tanh(inner));
        out[idx] = half(y);
    }
}
