#include <metal_stdlib>
using namespace metal;
inline float metalmind_gelu(float z) {
    float z2 = z * z;
    float inner = 0.7978845608028654f * (z + 0.044715f * z2 * z);
    return 0.5f * z * (1.0f + tanh(inner));
}
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint2 gid [[thread_position_in_grid]]) {
    (void)epsilon;
    if (gid.y >= B) {
        return;
    }
    uint col = gid.x << 2;
    if (col >= D) {
        return;
    }
    uint idx = gid.y * D + col;
    for (uint i = 0; i < 4; ++i) {
        uint c = col + i;
        if (c >= D) {
            break;
        }
        out[idx + i] = metalmind_gelu(x[idx + i] + parameter[c]);
    }
}
