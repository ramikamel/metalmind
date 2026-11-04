#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 gid [[thread_position_in_grid]]) {
    (void)epsilon;
    uint col = gid.x;
    uint rowBase = gid.y * 4u;
    if (col >= D || rowBase >= B) {
        return;
    }
    float bias = float(parameter[col]);
    for (uint r = 0u; r < 4u; ++r) {
        uint row = rowBase + r;
        if (row >= B) {
            break;
        }
        uint idx = row * D + col;
        float z = float(x[idx]) + bias;
        float z2 = z * z;
        float inner = 0.7978845608028654f * (z + 0.044715f * z2 * z);
        float y = 0.5f * z * (1.0f + tanh(inner));
        out[idx] = float(y);
    }
}
