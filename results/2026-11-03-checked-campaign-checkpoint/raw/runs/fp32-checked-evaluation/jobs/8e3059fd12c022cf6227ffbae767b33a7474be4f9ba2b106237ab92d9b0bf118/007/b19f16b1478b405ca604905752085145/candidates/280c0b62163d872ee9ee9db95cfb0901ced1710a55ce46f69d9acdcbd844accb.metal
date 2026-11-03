#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tg [[threadgroup_position_in_grid]], uint3 tid3 [[thread_position_in_threadgroup]], uint3 tpg [[threads_per_threadgroup]]) {
  uint row = tg.x;
  if (row >= B) return;
  uint tid = tid3.x;
  uint nt = tpg.x;
  threadgroup float shared[256];
  device const float* xrow = x + (uint)((uint)row * D);
  float sum = 0.0f;
  if ((D & 3u) == 0u) {
    device const float4* x4 = (device const float4*)xrow;
    uint d4 = D >> 2;
    for (uint i = tid; i < d4; i += nt) {
      float4 v = x4[i];
      sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
  } else {
    for (uint i = tid; i < D; i += nt) {
      float v = xrow[i];
      sum += v * v;
    }
  }
  if (tid < 256u) shared[tid] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = nt >> 1; s > 0u; s >>= 1) {
    if (tid < s) shared[tid] += shared[tid + s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0u) {
    float mean = shared[0] / (float)D;
    shared[0] = rsqrt(mean + epsilon);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float inv = shared[0];
  device float* orow = out + (uint)((uint)row * D);
  if ((D & 3u) == 0u) {
    device const float4* x4 = (device const float4*)xrow;
    device const float4* p4 = (device const float4*)parameter;
    device float4* o4 = (device float4*)orow;
    uint d4 = D >> 2;
    for (uint i = tid; i < d4; i += nt) {
      float4 v = x4[i];
      float4 p = p4[i];
      float4 r;
      r.x = v.x * inv * p.x;
      r.y = v.y * inv * p.y;
      r.z = v.z * inv * p.z;
      r.w = v.w * inv * p.w;
      o4[i] = r;
    }
  } else {
    for (uint i = tid; i < D; i += nt) {
      orow[i] = xrow[i] * inv * parameter[i];
    }
  }
}
