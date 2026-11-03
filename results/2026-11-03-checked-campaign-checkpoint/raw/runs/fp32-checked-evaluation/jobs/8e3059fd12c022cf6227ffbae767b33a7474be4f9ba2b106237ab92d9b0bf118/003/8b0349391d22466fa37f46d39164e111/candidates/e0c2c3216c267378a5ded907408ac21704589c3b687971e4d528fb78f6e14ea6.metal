#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint2 gid [[thread_position_in_grid]]) {
    (void)epsilon;
    uint col = gid.x;
    uint row = gid.y;
    if (row >= B || col >= D) return;
    uint idx = row * D + col;
    float z = float(x[idx]) + float(parameter[col]);
    float z2 = z * z;
    float inner = 0.7978845608028654f * (z + 0.044715f * z2 * z);
    float y = 0.5f * z * (1.0f + tanh(inner));
    out[idx] = float(y);
}
