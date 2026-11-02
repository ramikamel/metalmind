#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tg [[threadgroup_position_in_grid]], uint3 tid3 [[thread_position_in_threadgroup]], uint3 tpg [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint nsg [[simdgroups_per_threadgroup]]) {
  uint row = tg.x;
  if (row >= B) return;
  uint tid = tid3.x;
  uint T = tpg.x;
  device const float* xr = x + row * D;
  device float* orow = out + row * D;
  if (D == 4096u && T == 256u) {
    threadgroup float red[32];
    threadgroup float scale_s;
    device const float4* xr4 = (device const float4*)xr;
    device float4* orow4 = (device float4*)orow;
    device const float4* prm4 = (device const float4*)parameter;
    float4 v0 = xr4[tid];
    float4 v1 = xr4[tid + 256u];
    float4 v2 = xr4[tid + 512u];
    float4 v3 = xr4[tid + 768u];
    float sum = v0.x * v0.x + v0.y * v0.y + v0.z * v0.z + v0.w * v0.w + v1.x * v1.x + v1.y * v1.y + v1.z * v1.z + v1.w * v1.w + v2.x * v2.x + v2.y * v2.y + v2.z * v2.z + v2.w * v2.w + v3.x * v3.x + v3.y * v3.y + v3.z * v3.z + v3.w * v3.w;
    sum = simd_sum(sum);
    if (lane == 0u) red[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
      float p = (lane < nsg) ? red[lane] : 0.0f;
      p = simd_sum(p);
      if (lane == 0u) scale_s = rsqrt(p * 0.000244140625f + epsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = scale_s;
    float4 p0 = prm4[tid];
    float4 p1 = prm4[tid + 256u];
    float4 p2 = prm4[tid + 512u];
    float4 p3 = prm4[tid + 768u];
    orow4[tid] = v0 * s * p0;
    orow4[tid + 256u] = v1 * s * p1;
    orow4[tid + 512u] = v2 * s * p2;
    orow4[tid + 768u] = v3 * s * p3;
    return;
  }
  threadgroup float partial[1024];
  threadgroup float scale_s;
  float sum = 0.0f;
  if ((D & 3u) == 0u) {
    uint D4 = D >> 2;
    device const float4* xr4 = (device const float4*)xr;
    for (uint i = tid; i < D4; i += T) { float4 v = xr4[i]; sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w; }
  } else {
    for (uint c = tid; c < D; c += T) { float v = xr[c]; sum += v * v; }
  }
  partial[tid] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint offset = T >> 1; offset > 0u; offset >>= 1) {
    if (tid < offset) partial[tid] += partial[tid + offset];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0u) scale_s = rsqrt(partial[0] / float(D) + epsilon);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float s = scale_s;
  if ((D & 3u) == 0u) {
    uint D4 = D >> 2;
    device const float4* xr4 = (device const float4*)xr;
    device float4* orow4 = (device float4*)orow;
    device const float4* prm4 = (device const float4*)parameter;
    for (uint i = tid; i < D4; i += T) { orow4[i] = xr4[i] * s * prm4[i]; }
  } else {
    for (uint c = tid; c < D; c += T) { orow[c] = xr[c] * s * parameter[c]; }
  }
}
