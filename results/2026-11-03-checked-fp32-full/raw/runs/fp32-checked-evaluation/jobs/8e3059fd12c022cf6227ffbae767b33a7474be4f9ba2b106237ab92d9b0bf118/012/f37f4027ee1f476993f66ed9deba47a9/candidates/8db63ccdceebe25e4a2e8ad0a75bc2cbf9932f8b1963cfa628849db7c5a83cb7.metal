#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint tid [[thread_position_in_threadgroup]], uint tgid [[threadgroup_position_in_grid]], uint tgsize [[threads_per_threadgroup]]) {
  (void)epsilon;
  threadgroup float red[128];
  uint row = tgid;
  if (row >= B) return;
  uint base = row * D;
  float local_max = -3.402823e38f;
  for (uint i = tid; i < D; i += tgsize) {
    float v = x[base + i];
    local_max = max(local_max, v);
  }
  red[tid] = local_max;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 64u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      red[tid] = max(red[tid], red[tid + stride]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float row_max = red[0];
  float local_sum = 0.0f;
  for (uint i = tid; i < D; i += tgsize) {
    local_sum += exp(x[base + i] - row_max);
  }
  red[tid] = local_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 64u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      red[tid] += red[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float inv_sum = 1.0f / red[0];
  for (uint i = tid; i < D; i += tgsize) {
    float e = exp(x[base + i] - row_max);
    out[base + i] = float(e * inv_sum);
  }
}
