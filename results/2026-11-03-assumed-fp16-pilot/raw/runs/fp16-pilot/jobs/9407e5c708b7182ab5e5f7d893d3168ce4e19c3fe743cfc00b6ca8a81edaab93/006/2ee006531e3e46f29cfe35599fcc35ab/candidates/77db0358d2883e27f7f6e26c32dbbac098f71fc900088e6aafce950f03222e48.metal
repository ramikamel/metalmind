#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device const half* parameter, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint2 pos [[thread_position_in_grid]], uint2 grid [[threads_per_grid]]) {
  (void)epsilon;
  if ((D & 15u) == 0u) {
    uint dv = D >> 2;
    device const half4* xv = reinterpret_cast<device const half4*>(x);
    device const half4* pv = reinterpret_cast<device const half4*>(parameter);
    device half4* ov = reinterpret_cast<device half4*>(out);
    uint row = pos.y;
    if (row < B) {
      uint c = pos.x << 2;
      if (c + 3u < dv) {
        uint base = row * dv + c;
        float4 p0 = float4(pv[c]);
        float4 p1 = float4(pv[c + 1u]);
        float4 p2 = float4(pv[c + 2u]);
        float4 p3 = float4(pv[c + 3u]);
        float4 z0 = float4(xv[base]) + p0;
        float4 z1 = float4(xv[base + 1u]) + p1;
        float4 z2v = float4(xv[base + 2u]) + p2;
        float4 z3 = float4(xv[base + 3u]) + p3;
        float4 i0 = 0.7978845608028654f * (z0 + 0.044715f * z0 * z0 * z0);
        float4 i1 = 0.7978845608028654f * (z1 + 0.044715f * z1 * z1 * z1);
        float4 i2 = 0.7978845608028654f * (z2v + 0.044715f * z2v * z2v * z2v);
        float4 i3 = 0.7978845608028654f * (z3 + 0.044715f * z3 * z3 * z3);
        float4 y0 = 0.5f * z0 * (1.0f + tanh(i0));
        float4 y1 = 0.5f * z1 * (1.0f + tanh(i1));
        float4 y2 = 0.5f * z2v * (1.0f + tanh(i2));
        float4 y3 = 0.5f * z3 * (1.0f + tanh(i3));
        ov[base] = half4(y0);
        ov[base + 1u] = half4(y1);
        ov[base + 2u] = half4(y2);
        ov[base + 3u] = half4(y3);
      } else {
        for (uint k = c; k < dv; ++k) {
          float4 z = float4(xv[row * dv + k]) + float4(pv[k]);
          float4 inner = 0.7978845608028654f * (z + 0.044715f * z * z * z);
          ov[row * dv + k] = half4(0.5f * z * (1.0f + tanh(inner)));
        }
      }
    }
  } else {
    uint total = B * D;
    uint lin = pos.y * grid.x + pos.x;
    uint stride = grid.x * grid.y;
    for (uint i = lin; i < total; i += stride) {
      uint col = (D == 0u) ? 0u : (i % D);
      float z = float(x[i]) + float(parameter[col]);
      float inner = 0.7978845608028654f * (z + 0.044715f * z * z * z);
      out[i] = half(0.5f * z * (1.0f + tanh(inner)));
    }
  }
}
