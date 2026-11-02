#include <metal_stdlib>
using namespace metal;
static float4 gelu4(float4 z) {
  float4 z3 = z * z * z;
  float4 inner = 0.7978845608028654f * (z + 0.044715f * z3);
  return 0.5f * z * (1.0f + tanh(inner));
}
static float gelu1(float z) {
  float z3 = z * z * z;
  float inner = 0.7978845608028654f * (z + 0.044715f * z3);
  return 0.5f * z * (1.0f + tanh(inner));
}
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint2 pos [[thread_position_in_grid]], uint2 grid [[threads_per_grid]]) {
  (void)epsilon;
  (void)grid;
  uint row = pos.y;
  if (row >= B) return;
  uint baseCol = pos.x * 16u;
  if (baseCol >= D) return;
  uint rowOffset = row * D;
  if ((D & 3u) == 0u && baseCol + 15u < D) {
    device const float4* x4 = (device const float4*)(x + rowOffset + baseCol);
    device const float4* p4 = (device const float4*)(parameter + baseCol);
    device float4* o4 = (device float4*)(out + rowOffset + baseCol);
    float4 p0 = p4[0];
    float4 p1 = p4[1];
    float4 p2 = p4[2];
    float4 p3 = p4[3];
    float4 z0 = x4[0] + p0;
    float4 z1 = x4[1] + p1;
    float4 z2 = x4[2] + p2;
    float4 z3v = x4[3] + p3;
    o4[0] = gelu4(z0);
    o4[1] = gelu4(z1);
    o4[2] = gelu4(z2);
    o4[3] = gelu4(z3v);
  } else {
    uint end = min(baseCol + 16u, D);
    for (uint c = baseCol; c < end; ++c) {
      float z = x[rowOffset + c] + parameter[c];
      out[rowOffset + c] = gelu1(z);
    }
  }
}
