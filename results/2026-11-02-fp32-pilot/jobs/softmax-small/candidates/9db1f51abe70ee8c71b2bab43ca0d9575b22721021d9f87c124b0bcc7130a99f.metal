#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 gid [[thread_position_in_grid]], uint3 tg [[thread_position_in_threadgroup]], uint3 tpg [[threads_per_threadgroup]]) {
  uint row = gid.x;
  if (row >= B) { return; }
  uint base = row * D;
  float row_max = -3.4028235e38f;
  for (uint i = 0; i < D; ++i) {
    row_max = max(row_max, x[base + i]);
  }
  float sum = 0.0f;
  for (uint i = 0; i < D; ++i) {
    sum += exp(x[base + i] - row_max);
  }
  float inv_sum = 1.0f / sum;
  for (uint i = 0; i < D; ++i) {
    out[base + i] = exp(x[base + i] - row_max) * inv_sum;
  }
}
