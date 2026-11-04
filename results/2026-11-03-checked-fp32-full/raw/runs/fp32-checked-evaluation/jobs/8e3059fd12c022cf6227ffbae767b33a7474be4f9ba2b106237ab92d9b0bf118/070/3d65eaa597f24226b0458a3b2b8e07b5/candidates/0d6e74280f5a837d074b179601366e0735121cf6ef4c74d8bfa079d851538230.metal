#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 thread_position_in_grid [[thread_position_in_grid]], uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]], uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]) {
  uint row = threadgroup_position_in_grid.x;
  if (row >= B) return;
  uint tid = thread_position_in_threadgroup.x;
  const uint GT = 256;
  device const float* xr = x + uint(row * D);
  device float* orow = out + uint(row * D);
  float sum = 0.0f;
  uint D4 = D / 4;
  device const float4* x4 = (device const float4*)xr;
  for (uint i = tid; i < D4; i += GT) {
    float4 v = x4[i];
    sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  }
  uint rem = D - D4 * 4;
  if (tid < rem) {
    float v = xr[D4 * 4 + tid];
    sum += v * v;
  }
  threadgroup float partial[256];
  threadgroup float shared_scale;
  partial[tid] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 128) partial[tid] += partial[tid + 128];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 64) partial[tid] += partial[tid + 64];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 32) partial[tid] += partial[tid + 32];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 16) partial[tid] += partial[tid + 16];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 8) partial[tid] += partial[tid + 8];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 4) partial[tid] += partial[tid + 4];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 2) partial[tid] += partial[tid + 2];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float total = partial[0] + partial[1];
    shared_scale = rsqrt(total / float(D) + epsilon);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float scale = shared_scale;
  device const float4* p4 = (device const float4*)parameter;
  device float4* o4 = (device float4*)orow;
  for (uint i = tid; i < D4; i += GT) {
    float4 v = x4[i];
    float4 p = p4[i];
    o4[i] = v * scale * p;
  }
  if (tid < rem) {
    uint idx = D4 * 4 + tid;
    orow[idx] = xr[idx] * scale * parameter[idx];
  }
}
