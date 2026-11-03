#include <metal_stdlib>
using namespace metal;
static float gelu_tanh(float z) {
    float z2 = z * z;
    float u = 0.7978845608028654f * (z + 0.044715f * z2 * z);
    float au = min(fabs(u), 12.0f);
    float e = exp(-2.0f * au);
    float t = (1.0f - e) / (1.0f + e);
    t = (u >= 0.0f) ? t : -t;
    return 0.5f * z * (1.0f + t);
}
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint gid [[thread_position_in_grid]]) {
    (void)epsilon;
    uint n = B * D;
    uint base = gid * 4u;
    if (base >= n) {
        return;
    }
    uint row = base / D;
    uint col = base - row * D;
    if (base + 3u < n) {
        float4 xv = ((device const float4*)x)[gid];
        uint c0 = col;
        uint c1 = col + 1u;
        uint c2 = col + 2u;
        uint c3 = col + 3u;
        c1 = (c1 >= D) ? (c1 - D) : c1;
        c2 = (c2 >= D) ? (c2 - D) : c2;
        c3 = (c3 >= D) ? (c3 - D) : c3;
        float4 y;
        y.x = gelu_tanh(xv.x + parameter[c0]);
        y.y = gelu_tanh(xv.y + parameter[c1]);
        y.z = gelu_tanh(xv.z + parameter[c2]);
        y.w = gelu_tanh(xv.w + parameter[c3]);
        ((device float4*)out)[gid] = y;
    } else {
        for (uint i = base; i < n; ++i) {
            uint r = i / D;
            uint c = i - r * D;
            out[i] = gelu_tanh(x[i] + parameter[c]);
        }
    }
}
