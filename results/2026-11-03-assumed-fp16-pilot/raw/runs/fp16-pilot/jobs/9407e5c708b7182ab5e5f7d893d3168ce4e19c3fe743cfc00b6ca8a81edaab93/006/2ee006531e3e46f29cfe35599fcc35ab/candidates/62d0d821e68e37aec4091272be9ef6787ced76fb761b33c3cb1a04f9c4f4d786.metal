#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device const half* parameter, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint2 pos [[thread_position_in_grid]], uint2 grid [[threads_per_grid]]) {
  (void)epsilon;
  if ((D & 3u) == 0u) {
    uint dv = D >> 2;
    device const half4* xv = reinterpret_cast<device const half4*>(x);
    device const half4* pv = reinterpret_cast<device const half4*>(parameter);
    device half4* ov = reinterpret_cast<device half4*>(out);
    for (uint row = pos.y; row < B; row += grid.y) {
      uint base = row * dv;
      for (uint col = pos.x; col < dv; col += grid.x) {
        float4 z = float4(xv[base + col]) + float4(pv[col]);
        float4 z2 = z * z;
        float4 inner = 0.7978845608028654f * (z + 0.044715f * z2 * z);
        float4 y = 0.5f * z * (1.0f + tanh(inner));
        ov[base + col] = half4(y);
      }
    }
  } else {
    uint total = B * D;
    uint lin = pos.y * grid.x + pos.x;
    uint stride = grid.x * grid.y;
    for (uint i = lin; i < total; i += stride) {
      uint col = (D == 0u) ? 0u : (i % D);
      float z = float(x[i]) + float(parameter[col]);
      float z2 = z * z;
      float inner = 0.7978845608028654f * (z + 0.044715f * z2 * z);
      float y = 0.5f * z * (1.0f + tanh(inner));
      out[i] = half(y);
    }
  }
}
