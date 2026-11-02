#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 gid [[thread_position_in_grid]], uint3 tg [[thread_position_in_threadgroup]], uint3 tpg [[threads_per_threadgroup]]) {
  uint row = gid.y;
  if (row >= B) { return; }
  uint lane = tg.x;
  uint nt = tpg.x;
  threadgroup float buf[256];
  uint base = row * D;
  float local_max = -3.4028235e38f;
  for (uint i = lane; i < D; i += nt) {
    local_max = max(local_max, x[base + i]);
  }
  buf[lane] = local_max;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = nt / 2; stride > 0; stride >>= 1) {
    if (lane < stride) { buf[lane] = max(buf[lane], buf[lane + stride]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float row_max = buf[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float local_sum = 0.0f;
  for (uint i = lane; i < D; i += nt) {
    local_sum += exp(x[base + i] - row_max);
  }
  buf[lane] = local_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = nt / 2; stride > 0; stride >>= 1) {
    if (lane < stride) { buf[lane] += buf[lane + stride]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float inv_sum = 1.0f / buf[0];
  for (uint i = lane; i < D; i += nt) {
    float v = exp(x[base + i] - row_max) * inv_sum;
    out[base + i] = v;
  }
}
