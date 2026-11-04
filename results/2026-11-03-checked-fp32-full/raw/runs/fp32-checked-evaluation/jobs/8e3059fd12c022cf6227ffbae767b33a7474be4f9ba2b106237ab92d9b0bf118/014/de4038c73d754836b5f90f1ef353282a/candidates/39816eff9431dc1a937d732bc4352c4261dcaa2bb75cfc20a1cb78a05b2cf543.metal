#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint gid [[thread_position_in_grid]]) {
    (void)epsilon;
    uint total = B * D;
    if (gid >= total) {
        return;
    }
    uint col = gid % D;
    float z = float(x[gid]) + float(parameter[col]);
    float z3 = z * z * z;
    float inner = 0.7978845608028654f * (z + 0.044715f * z3);
    float y = 0.5f * z * (1.0f + tanh(inner));
    out[gid] = float(y);
}
