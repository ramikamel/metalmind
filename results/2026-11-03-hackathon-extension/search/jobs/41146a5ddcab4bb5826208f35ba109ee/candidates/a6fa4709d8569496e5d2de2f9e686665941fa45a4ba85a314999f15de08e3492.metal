#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint tid [[thread_index_in_threadgroup]], uint gid [[threadgroup_position_in_grid]], uint tgsize [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]]) {
  (void)epsilon;
  uint row = gid;
  if (row >= B) return;
  uint base = row * D;
  threadgroup float sh[32];
  bool vec = (D == 1024u && tgsize == 256u);
  float4 v = float4(0.0f);
  float lm = -3.402823466e+38f;
  if (vec) {
    device const float4* x4 = (device const float4*)(x + base);
    v = x4[tid];
    lm = max(max(v.x, v.y), max(v.z, v.w));
  } else {
    for (uint i = tid; i < D; i += tgsize) lm = max(lm, x[base + i]);
  }
  float sm = simd_max(lm);
  if (lane == 0u) sh[sg] = sm;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u) {
    float q = (lane < nsg) ? sh[lane] : -3.402823466e+38f;
    float gm = simd_max(q);
    if (lane == 0u) sh[0] = gm;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float rm = sh[0];
  float4 ev = float4(0.0f);
  float ls = 0.0f;
  if (vec) {
    ev = exp(v - rm);
    ls = (ev.x + ev.y) + (ev.z + ev.w);
  } else {
    for (uint i = tid; i < D; i += tgsize) ls += exp(x[base + i] - rm);
  }
  float ss = simd_sum(ls);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0u) sh[sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u) {
    float q = (lane < nsg) ? sh[lane] : 0.0f;
    float gs = simd_sum(q);
    if (lane == 0u) sh[0] = gs;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float inv = 1.0f / sh[0];
  if (vec) {
    device float4* o4 = (device float4*)(out + base);
    o4[tid] = ev * inv;
  } else {
    for (uint i = tid; i < D; i += tgsize) out[base + i] = exp(x[base + i] - rm) * inv;
  }
}
