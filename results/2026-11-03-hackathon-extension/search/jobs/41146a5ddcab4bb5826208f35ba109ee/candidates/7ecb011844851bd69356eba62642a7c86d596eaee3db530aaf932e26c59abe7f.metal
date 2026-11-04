#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint tid [[thread_index_in_threadgroup]], uint gid [[threadgroup_position_in_grid]], uint ngroups [[threadgroups_per_grid]], uint tgsize [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint simd_id [[simdgroup_index_in_threadgroup]], uint simd_size [[threads_per_simdgroup]]) {
  (void)epsilon;
  if (D == 1024u && simd_size == 32u && tgsize == 128u) {
    uint row = gid;
    if (row >= B) return;
    threadgroup float part[8];
    threadgroup float shared_rm;
    threadgroup float shared_inv;
    uint base = row * D;
    device const float4* x4 = (device const float4*)(x + base);
    float4 v0 = x4[tid];
    float4 v1 = x4[tid + 128u];
    float lm = max(max(v0.x, v0.y), max(v0.z, v0.w));
    lm = max(lm, max(max(v1.x, v1.y), max(v1.z, v1.w)));
    float pm = simd_max(lm);
    if (lane == 0u) part[simd_id] = pm;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0u) {
      float v = (lane < 4u) ? part[lane] : -3.402823466e+38f;
      float m = simd_max(v);
      if (lane == 0u) shared_rm = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rm = shared_rm;
    float c = 1.4426950408889634f;
    float4 e0 = exp2((v0 - rm) * c);
    float4 e1 = exp2((v1 - rm) * c);
    float ls = ((e0.x + e0.y) + (e0.z + e0.w)) + ((e1.x + e1.y) + (e1.z + e1.w));
    float ps = simd_sum(ls);
    if (lane == 0u) part[simd_id] = ps;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0u) {
      float v = (lane < 4u) ? part[lane] : 0.0f;
      float s = simd_sum(v);
      if (lane == 0u) shared_inv = 1.0f / s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = shared_inv;
    device float4* o4 = (device float4*)(out + base);
    o4[tid] = e0 * inv;
    o4[tid + 128u] = e1 * inv;
    return;
  }
  uint nsg = (tgsize + simd_size - 1u) / simd_size;
  uint row0 = gid * nsg + simd_id;
  uint rowStride = ngroups * nsg;
  for (uint row = row0; row < B; row += rowStride) {
    uint base = row * D;
    float lm = -3.402823466e+38f;
    for (uint i = lane; i < D; i += simd_size) lm = max(lm, x[base + i]);
    float rm = simd_max(lm);
    float ls = 0.0f;
    for (uint i = lane; i < D; i += simd_size) ls += exp(x[base + i] - rm);
    float inv = 1.0f / simd_sum(ls);
    for (uint i = lane; i < D; i += simd_size) out[base + i] = exp(x[base + i] - rm) * inv;
  }
}
