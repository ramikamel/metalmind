#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint tid [[thread_index_in_threadgroup]], uint gid [[threadgroup_position_in_grid]], uint ngroups [[threadgroups_per_grid]], uint tgsize [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint simd_id [[simdgroup_index_in_threadgroup]], uint simd_size [[threads_per_simdgroup]]) {
  (void)epsilon;
  threadgroup float sh[128];
  uint nsg = (tgsize + simd_size - 1u) / simd_size;
  const float log2e = 1.4426950408889634f;
  for (uint row = gid; row < B; row += ngroups) {
    uint base = row * D;
    if (D == 1024u && tgsize == 128u) {
      device const float4* x4 = (device const float4*)(x + base);
      float4 v0 = x4[tid];
      float4 v1 = x4[tid + 128u];
      float lm = max(max(v0.x, v0.y), max(v0.z, v0.w));
      lm = max(lm, max(max(v1.x, v1.y), max(v1.z, v1.w)));
      float sm = simd_max(lm);
      if (lane == 0u) sh[simd_id] = sm;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float rm = -3.402823466e+38f;
      for (uint i = 0u; i < nsg; ++i) rm = max(rm, sh[i]);
      float4 e0 = exp2((v0 - rm) * log2e);
      float4 e1 = exp2((v1 - rm) * log2e);
      float ls = ((e0.x + e0.y) + (e0.z + e0.w)) + ((e1.x + e1.y) + (e1.z + e1.w));
      float ss = simd_sum(ls);
      if (lane == 0u) sh[64u + simd_id] = ss;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float total = 0.0f;
      for (uint i = 0u; i < nsg; ++i) total += sh[64u + i];
      float inv = 1.0f / total;
      device float4* o4 = (device float4*)(out + base);
      o4[tid] = e0 * inv;
      o4[tid + 128u] = e1 * inv;
    } else {
      float lm = -3.402823466e+38f;
      for (uint i = tid; i < D; i += tgsize) lm = max(lm, x[base + i]);
      float sm = simd_max(lm);
      if (lane == 0u) sh[simd_id] = sm;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float rm = -3.402823466e+38f;
      for (uint i = 0u; i < nsg; ++i) rm = max(rm, sh[i]);
      float ls = 0.0f;
      for (uint i = tid; i < D; i += tgsize) ls += exp(x[base + i] - rm);
      float ss = simd_sum(ls);
      if (lane == 0u) sh[64u + simd_id] = ss;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float total = 0.0f;
      for (uint i = 0u; i < nsg; ++i) total += sh[64u + i];
      float inv = 1.0f / total;
      for (uint i = tid; i < D; i += tgsize) out[base + i] = exp(x[base + i] - rm) * inv;
    }
  }
}
