#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x [[buffer(0)]], device const float* parameter [[buffer(1)]], device float* out [[buffer(2)]], constant uint& B [[buffer(3)]], constant uint& D [[buffer(4)]], constant float& epsilon [[buffer(5)]], uint2 gid [[thread_position_in_grid]]) {
    (void)epsilon;
    if (gid.x >= D || gid.y >= B) return;
    uint idx = gid.y * D + gid.x;
    float z = float(x[idx]) + float(parameter[gid.x]);
    float z3 = z * z * z;
    float y = 0.5f * z * (1.0f + tanh(0.7978845608028654f * (z + 0.044715f * z3)));
    out[idx] = float(y);
}
