#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint gid [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tgid [[threadgroup_position_in_grid]]) {
  (void)gid;
  (void)epsilon;
  uint row = tgid;
  if (row >= B || D == 0) {
    return;
  }
  threadgroup float buf[256];
  uint base = row * D;
  float local_max = -3.4028235e38f;
  for (uint i = tid; i < D; i += 256) {
    local_max = max(local_max, x[base + i]);
  }
  buf[tid] = local_max;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
      buf[tid] = max(buf[tid], buf[tid + stride]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float row_max = buf[0];
  float local_sum = 0.0f;
  for (uint i = tid; i < D; i += 256) {
    local_sum += exp(x[base + i] - row_max);
  }
  buf[tid] = local_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
      buf[tid] += buf[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float inv_sum = 1.0f / buf[0];
  for (uint i = tid; i < D; i += 256) {
    out[base + i] = exp(x[base + i] - row_max) * inv_sum;
  }
}
