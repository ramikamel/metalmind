#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tg_pos [[threadgroup_position_in_grid]], uint3 tpig [[thread_position_in_threadgroup]], uint3 tg_size [[threads_per_threadgroup]], uint sg_id [[simdgroup_index_in_threadgroup]], uint lane_id [[thread_index_in_simdgroup]], uint sg_width [[threads_per_simdgroup]], uint sgs_per_tg [[simdgroups_per_threadgroup]]) {
  uint row = tg_pos.x;
  if (row >= B) { return; }
  uint tid = tpig.x;
  uint nthreads = tg_size.x;
  threadgroup float red[64];
  if (D == 1024u && nthreads == 256u) {
    uint base4 = row * (D / 4u);
    float4 v = ((device const float4*)x)[base4 + tid];
    float local_max = max(max(v.x, v.y), max(v.z, v.w));
    float simd_m = simd_max(local_max);
    if (lane_id == 0u) { red[sg_id] = simd_m; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = 0.0f;
    if (sg_id == 0u) {
      float m = (lane_id < sgs_per_tg) ? red[lane_id] : -3.402823466e+38f;
      row_max = simd_max(m);
      if (lane_id == 0u) { red[0] = row_max; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    row_max = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 e = exp(v - row_max);
    float local_sum = e.x + e.y + e.z + e.w;
    float simd_s = simd_sum(local_sum);
    if (lane_id == 0u) { red[sg_id] = simd_s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = 0.0f;
    if (sg_id == 0u) {
      float s = (lane_id < sgs_per_tg) ? red[lane_id] : 0.0f;
      float row_sum = simd_sum(s);
      inv_sum = 1.0f / row_sum;
      if (lane_id == 0u) { red[0] = inv_sum; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    inv_sum = red[0];
    ((device float4*)out)[base4 + tid] = e * inv_sum;
    return;
  }
  uint base = row * D;
  float local_max = -3.402823466e+38f;
  for (uint i = tid; i < D; i += nthreads) { local_max = max(local_max, x[base + i]); }
  float simd_m = simd_max(local_max);
  if (lane_id == 0u) { red[sg_id] = simd_m; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg_id == 0u) {
    float m = (lane_id < sgs_per_tg) ? red[lane_id] : -3.402823466e+38f;
    float rm = simd_max(m);
    if (lane_id == 0u) { red[0] = rm; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float row_max = red[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float local_sum = 0.0f;
  for (uint i = tid; i < D; i += nthreads) { local_sum += exp(x[base + i] - row_max); }
  float simd_s = simd_sum(local_sum);
  if (lane_id == 0u) { red[sg_id] = simd_s; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg_id == 0u) {
    float s = (lane_id < sgs_per_tg) ? red[lane_id] : 0.0f;
    float rs = simd_sum(s);
    if (lane_id == 0u) { red[0] = 1.0f / rs; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float inv_sum = red[0];
  for (uint i = tid; i < D; i += nthreads) { out[base + i] = exp(x[base + i] - row_max) * inv_sum; }
}
