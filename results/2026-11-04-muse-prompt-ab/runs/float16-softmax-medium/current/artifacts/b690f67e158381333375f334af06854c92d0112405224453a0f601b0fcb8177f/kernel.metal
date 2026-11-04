#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tg_id [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 tgs [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint spt [[threads_per_simdgroup]]) {
  threadgroup float red[256];
  (void)epsilon;
  uint row = tg_id.x;
  if (row >= B) { return; }
  uint lid = tid.x;
  uint width = tgs.x;
  if (width == 0u || lid >= 256u) { return; }
  device const half* rp = x + row * D;
  device half* op = out + row * D;
  uint nsg = (width + spt - 1u) / spt;
  if (D == width * 4u) {
    device const half4* rp4 = (device const half4*)rp;
    device half4* op4 = (device half4*)op;
    float4 v = float4(rp4[lid]);
    float lm = max(max(v.x, v.y), max(v.z, v.w));
    float m = simd_max(lm);
    if (lane == 0u) { red[sg] = m; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
      float mm = (lane < nsg) ? red[lane] : -3.4028235e38f;
      mm = simd_max(mm);
      if (lane == 0u) { red[0] = mm; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = red[0];
    float4 e = exp(v - row_max);
    float ls = (e.x + e.y) + (e.z + e.w);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = simd_sum(ls);
    if (lane == 0u) { red[sg] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
      float ss = (lane < nsg) ? red[lane] : 0.0f;
      ss = simd_sum(ss);
      if (lane == 0u) { red[0] = 1.0f / ss; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = red[0];
    op4[lid] = half4(e * inv_sum);
    return;
  }
  float local_max = -3.4028235e38f;
  for (uint i = lid; i < D; i += width) { local_max = max(local_max, float(rp[i])); }
  float m = simd_max(local_max);
  if (lane == 0u) { red[sg] = m; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u) {
    float mm = (lane < nsg) ? red[lane] : -3.4028235e38f;
    mm = simd_max(mm);
    if (lane == 0u) { red[0] = mm; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float row_max = red[0];
  float local_sum = 0.0f;
  for (uint i = lid; i < D; i += width) { local_sum += exp(float(rp[i]) - row_max); }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float s = simd_sum(local_sum);
  if (lane == 0u) { red[sg] = s; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u) {
    float ss = (lane < nsg) ? red[lane] : 0.0f;
    ss = simd_sum(ss);
    if (lane == 0u) { red[0] = 1.0f / ss; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float inv_sum = red[0];
  for (uint i = lid; i < D; i += width) { op[i] = half(exp(float(rp[i]) - row_max) * inv_sum); }
}
