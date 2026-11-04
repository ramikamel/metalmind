#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint row [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]], uint tgs [[threads_per_threadgroup]]) {
  (void)epsilon;
  if (row >= B) return;
  threadgroup float buf[256];
  uint base = row * D;
  float local_max = -3.4028235e38f;
  for (uint i = lid; i < D; i += tgs) {
    float v = x[base + i];
    local_max = max(local_max, v);
  }
  buf[lid] = local_max;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = tgs >> 1; stride > 0; stride >>= 1) {
    if (lid < stride) buf[lid] = max(buf[lid], buf[lid + stride]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float row_max = buf[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float local_sum = 0.0f;
  for (uint i = lid; i < D; i += tgs) {
    local_sum += exp(x[base + i] - row_max);
  }
  buf[lid] = local_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = tgs >> 1; stride > 0; stride >>= 1) {
    if (lid < stride) buf[lid] += buf[lid + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float inv_sum = 1.0f / buf[0];
  for (uint i = lid; i < D; i += tgs) {
    float v = exp(x[base + i] - row_max) * inv_sum;
    out[base + i] = v;
  }
}
