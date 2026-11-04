#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tg [[threadgroup_position_in_grid]], uint3 tid3 [[thread_position_in_threadgroup]], uint3 tpg [[threads_per_threadgroup]]) {
  uint row = tg.x;
  if (row >= B) { return; }
  uint tid = tid3.x;
  uint nthreads = tpg.x;
  device const float* xr = x + (uint)row * D;
  device float* outr = out + (uint)row * D;
  threadgroup float sdata[256];
  float local_max = -INFINITY;
  for (uint i = tid; i < D; i += nthreads) { local_max = max(local_max, xr[i]); }
  sdata[tid] = local_max;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = nthreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) { sdata[tid] = max(sdata[tid], sdata[tid + stride]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float row_max = sdata[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float local_sum = 0.0f;
  for (uint i = tid; i < D; i += nthreads) { local_sum += exp(xr[i] - row_max); }
  sdata[tid] = local_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = nthreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) { sdata[tid] += sdata[tid + stride]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float inv_sum = 1.0f / sdata[0];
  for (uint i = tid; i < D; i += nthreads) { outr[i] = exp(xr[i] - row_max) * inv_sum; }
}
