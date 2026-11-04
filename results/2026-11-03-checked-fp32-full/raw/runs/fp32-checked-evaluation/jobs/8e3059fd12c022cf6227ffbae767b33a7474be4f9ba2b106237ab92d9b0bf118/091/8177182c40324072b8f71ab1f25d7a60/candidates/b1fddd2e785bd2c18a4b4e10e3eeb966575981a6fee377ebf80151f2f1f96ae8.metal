#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint gid [[thread_position_in_grid]]) {
  (void)epsilon;
  uint total = B * D;
  uint base = gid * 4u;
  if (base >= total) return;
  const float k = 0.7978845608028654f;
  const float c = -2.8853900817779268f;
  uint col = (D == 0u) ? 0u : (base % D);
  if (base + 3u < total && col + 3u < D && (col & 3u) == 0u) {
    float4 xv = *(device const float4*)(x + base);
    float4 pv = *(device const float4*)(parameter + col);
    float4 z = xv + pv;
    float4 cube = z * z * z;
    float4 u = k * fma(0.044715f, cube, z);
    float4 s = 1.0f / (1.0f + exp2(c * u));
    *(device float4*)(out + base) = z * s;
  } else {
    for (uint i = 0u; i < 4u; ++i) {
      uint idx = base + i;
      if (idx >= total) return;
      uint cc = (D == 0u) ? 0u : (idx % D);
      float z = x[idx] + parameter[cc];
      float cube = z * z * z;
      float u = k * fma(0.044715f, cube, z);
      float s = 1.0f / (1.0f + exp2(c * u));
      out[idx] = z * s;
    }
  }
}
