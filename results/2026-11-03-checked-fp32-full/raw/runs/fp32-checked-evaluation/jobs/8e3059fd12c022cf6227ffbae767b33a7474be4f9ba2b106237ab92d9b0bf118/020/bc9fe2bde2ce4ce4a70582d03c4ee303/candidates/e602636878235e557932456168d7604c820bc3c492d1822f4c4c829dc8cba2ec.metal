#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 gid [[thread_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 tgid [[threadgroup_position_in_grid]], uint3 tgdim [[threadgroup_dimensions]]) {
  uint row = tgid.x;
  if (row >= B) return;
  uint lid = tid.x;
  uint nthreads = tgdim.x;
  device const float* row_x = x + (uint)row * D;
  device float* row_o = out + (uint)row * D;
  float local = 0.0f;
  bool vec = ((D & 3u) == 0u);
  if (vec) {
    device const float4* xv = (device const float4*)row_x;
    uint D4 = D >> 2;
    for (uint i = lid; i < D4; i += nthreads) {
      float4 v = xv[i];
      local += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
  } else {
    for (uint i = lid; i < D; i += nthreads) {
      float v = row_x[i];
      local += v * v;
    }
  }
  threadgroup float shared[256];
  shared[lid] = local;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = nthreads >> 1; stride > 0u; stride >>= 1) {
    if (lid < stride) shared[lid] += shared[lid + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float scale = rsqrt(shared[0] / (float)D + epsilon);
  if (vec) {
    device const float4* xv = (device const float4*)row_x;
    device float4* ov = (device float4*)row_o;
    device const float4* pv = (device const float4*)parameter;
    uint D4 = D >> 2;
    for (uint i = lid; i < D4; i += nthreads) {
      float4 v = xv[i];
      float4 p = pv[i];
      float4 r;
      r.x = v.x * scale * p.x;
      r.y = v.y * scale * p.y;
      r.z = v.z * scale * p.z;
      r.w = v.w * scale * p.w;
      ov[i] = r;
    }
  } else {
    for (uint i = lid; i < D; i += nthreads) {
      row_o[i] = row_x[i] * scale * parameter[i];
    }
  }
}
