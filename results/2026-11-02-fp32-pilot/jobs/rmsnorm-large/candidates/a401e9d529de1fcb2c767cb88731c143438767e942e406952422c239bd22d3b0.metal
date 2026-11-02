#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tg [[threadgroup_position_in_grid]], uint3 tid3 [[thread_position_in_threadgroup]], uint3 tpg [[threads_per_threadgroup]]) {
  uint row = tg.x;
  if (row >= B) return;
  uint tid = tid3.x;
  uint T = tpg.x;
  threadgroup float partial[1024];
  threadgroup float scale_s;
  device const float* xr = x + row * D;
  device float* orow = out + row * D;
  float sum = 0.0f;
  if ((D & 3u) == 0u) {
    uint D4 = D >> 2;
    device const float4* xr4 = (device const float4*)xr;
    for (uint i = tid; i < D4; i += T) {
      float4 v = xr4[i];
      sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
  } else {
    for (uint c = tid; c < D; c += T) {
      float v = xr[c];
      sum += v * v;
    }
  }
  partial[tid] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint offset = T >> 1; offset > 0u; offset >>= 1) {
    if (tid < offset) partial[tid] += partial[tid + offset];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0u) {
    scale_s = rsqrt(partial[0] / float(D) + epsilon);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float s = scale_s;
  if ((D & 3u) == 0u) {
    uint D4 = D >> 2;
    device const float4* xr4 = (device const float4*)xr;
    device float4* orow4 = (device float4*)orow;
    device const float4* prm4 = (device const float4*)parameter;
    for (uint i = tid; i < D4; i += T) {
      float4 v = xr4[i];
      float4 p = prm4[i];
      orow4[i] = v * s * p;
    }
  } else {
    for (uint c = tid; c < D; c += T) {
      orow[c] = xr[c] * s * parameter[c];
    }
  }
}
