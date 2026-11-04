#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tg_pos [[threadgroup_position_in_grid]], uint sg_id [[simdgroup_index_in_threadgroup]], uint lane_id [[thread_index_in_simdgroup]], uint sg_width [[threads_per_simdgroup]], uint sgs_per_tg [[simdgroups_per_threadgroup]]) {
  uint row = tg_pos.x * sgs_per_tg + sg_id;
  if (row >= B) { return; }
  uint base = row * D;
  if ((D % (sg_width * 4u)) == 0u && ((D / 4u) / sg_width) == 8u) {
    uint base4 = row * (D / 4u);
    float4 vals[8];
    float local_max = -3.402823466e+38f;
    for (uint k = 0u; k < 8u; ++k) {
      uint idx = lane_id + k * sg_width;
      float4 v = ((device const float4*)x)[base4 + idx];
      vals[k] = v;
      local_max = max(local_max, max(max(v.x, v.y), max(v.z, v.w)));
    }
    float row_max = simd_max(local_max);
    float local_sum = 0.0f;
    for (uint k = 0u; k < 8u; ++k) {
      float4 e = exp(vals[k] - row_max);
      vals[k] = e;
      local_sum += e.x + e.y + e.z + e.w;
    }
    float row_sum = simd_sum(local_sum);
    float inv_sum = 1.0f / row_sum;
    for (uint k = 0u; k < 8u; ++k) {
      uint idx = lane_id + k * sg_width;
      ((device float4*)out)[base4 + idx] = vals[k] * inv_sum;
    }
    return;
  }
  float local_max = -3.402823466e+38f;
  for (uint i = lane_id; i < D; i += sg_width) {
    local_max = max(local_max, x[base + i]);
  }
  float row_max = simd_max(local_max);
  float local_sum = 0.0f;
  for (uint i = lane_id; i < D; i += sg_width) {
    local_sum += exp(x[base + i] - row_max);
  }
  float row_sum = simd_sum(local_sum);
  float inv_sum = 1.0f / row_sum;
  for (uint i = lane_id; i < D; i += sg_width) {
    out[base + i] = exp(x[base + i] - row_max) * inv_sum;
  }
}
