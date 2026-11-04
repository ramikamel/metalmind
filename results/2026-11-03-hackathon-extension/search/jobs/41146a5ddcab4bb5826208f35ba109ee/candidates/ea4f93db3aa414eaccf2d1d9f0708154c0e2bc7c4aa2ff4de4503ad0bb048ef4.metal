#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint tid [[thread_index_in_threadgroup]], uint gid [[threadgroup_position_in_grid]], uint tgsize [[threads_per_threadgroup]]) {
  (void)epsilon;
  (void)tgsize;
  uint row = gid;
  if (row >= B) return;
  uint base = row * D;
  threadgroup float sh[256];
  bool vec = (D == 1024u);
  float4 v = float4(0.0f);
  float lm = -3.402823466e+38f;
  if (vec) {
    device const float4* x4 = (device const float4*)(x + base);
    v = x4[tid];
    lm = max(max(v.x, v.y), max(v.z, v.w));
  } else {
    for (uint i = tid; i < D; i += 256u) lm = max(lm, x[base + i]);
  }
  sh[tid] = lm;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = 128u; s > 0u; s >>= 1u) {
    if (tid < s) sh[tid] = max(sh[tid], sh[tid + s]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float rm = sh[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float4 ev = float4(0.0f);
  float ls = 0.0f;
  if (vec) {
    ev = exp(v - rm);
    ls = (ev.x + ev.y) + (ev.z + ev.w);
  } else {
    for (uint i = tid; i < D; i += 256u) ls += exp(x[base + i] - rm);
  }
  sh[tid] = ls;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = 128u; s > 0u; s >>= 1u) {
    if (tid < s) sh[tid] += sh[tid + s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float inv = 1.0f / sh[0];
  if (vec) {
    device float4* o4 = (device float4*)(out + base);
    o4[tid] = ev * inv;
  } else {
    for (uint i = tid; i < D; i += 256u) out[base + i] = exp(x[base + i] - rm) * inv;
  }
}