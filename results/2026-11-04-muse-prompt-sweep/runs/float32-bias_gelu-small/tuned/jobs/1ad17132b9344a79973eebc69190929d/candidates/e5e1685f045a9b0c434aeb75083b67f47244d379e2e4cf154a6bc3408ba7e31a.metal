#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint2 gid [[thread_position_in_grid]], uint2 grid_dim [[threads_per_grid]]) {
    if (gid.y >= B) return;
    uint row_offset = gid.y * D;
    for (uint col = gid.x; col < D; col += grid_dim.x) {
        float z = x[row_offset + col] + parameter[col];
        float z2 = z * z;
        float u = 0.7978845608028654f * (z + 0.044715f * z * z2);
        out[row_offset + col] = 0.5f * z * (1.0f + tanh(u));
    }
}
