#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint gid [[thread_position_in_grid]]) {
  (void)epsilon;
  uint total = B * D;
  uint base = gid * 4u;
  if (base >= total) return;
  const float k = 0.7978845608028654f;
  uint col = (D == 0u) ? 0u : (base % D);
  if (base + 3u < total && col + 3u < D) {
    float4 xv = *(device const float4*)(x + base);
    float4 pv = *(device const float4*)(parameter + col);
    float4 z = xv + pv;
    float4 cube = z * z * z;
    float4 inner = k * (z + 0.044715f * cube);
    float4 y = 0.5f * z * (1.0f + tanh(inner));
    *(device float4*)(out + base) = y;
  } else {
    for (uint i = 0u; i < 4u; ++i) {
      uint idx = base + i;
      if (idx >= total) return;
      uint c = (D == 0u) ? 0u : (idx % D);
      float z = x[idx] + parameter[c];
      float cube = z * z * z;
      float y = 0.5f * z * (1.0f + tanh(k * (z + 0.044715f * cube)));
      out[idx] = y;
    }
  }
}
