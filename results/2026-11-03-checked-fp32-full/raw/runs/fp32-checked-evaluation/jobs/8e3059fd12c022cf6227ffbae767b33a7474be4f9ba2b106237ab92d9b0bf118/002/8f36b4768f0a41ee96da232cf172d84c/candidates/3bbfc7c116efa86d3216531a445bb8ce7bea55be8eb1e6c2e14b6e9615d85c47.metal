#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 gid [[thread_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 tpg [[threads_per_threadgroup]], uint3 tgpos [[threadgroup_position_in_grid]]) {
  (void)gid;
  (void)epsilon;
  uint row = tgpos.y;
  if (row >= B) return;
  uint lane = tid.x;
  uint nthreads = tpg.x;
  device const float* row_in = x + (uint)(row * D);
  device float* row_out = out + (uint)(row * D);
  threadgroup float red[256];
  threadgroup float cache[4096];
  bool use_cache = (D <= 4096u);
  float local_max = -3.4028235e38f;
  for (uint i = lane; i < D; i += nthreads) {
    local_max = max(local_max, row_in[i]);
  }
  red[lane] = local_max;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = nthreads >> 1; stride > 0u; stride >>= 1) {
    if (lane < stride) red[lane] = max(red[lane], red[lane + stride]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float row_max = red[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float local_sum = 0.0f;
  if (use_cache) {
    for (uint i = lane; i < D; i += nthreads) {
      float e = exp(row_in[i] - row_max);
      cache[i] = e;
      local_sum += e;
    }
  } else {
    for (uint i = lane; i < D; i += nthreads) {
      local_sum += exp(row_in[i] - row_max);
    }
  }
  red[lane] = local_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = nthreads >> 1; stride > 0u; stride >>= 1) {
    if (lane < stride) red[lane] += red[lane + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float inv_sum = 1.0f / red[0];
  if (use_cache) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lane; i < D; i += nthreads) {
      row_out[i] = (float)(cache[i] * inv_sum);
    }
  } else {
    for (uint i = lane; i < D; i += nthreads) {
      row_out[i] = (float)(exp(row_in[i] - row_max) * inv_sum);
    }
  }
}
