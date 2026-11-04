#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint2 gid [[thread_position_in_grid]]) {
    uint col = gid.x;
    if (col >= D) return;
    float p = float(parameter[col]);
    uint row0 = gid.y * 8u;
    if (row0 >= B) return;
    uint rows = min(8u, B - row0);
    uint idx = row0 * D + col;
    for (uint i = 0u; i < rows; ++i) {
        float z = float(x[idx]) + p;
        float z2 = z * z;
        float inner = 0.7978845608028654f * fma(0.044715f, z * z2, z);
        float y = 0.5f * z * (1.0f + tanh(inner));
        out[idx] = float(y);
        idx += D;
    }
}
