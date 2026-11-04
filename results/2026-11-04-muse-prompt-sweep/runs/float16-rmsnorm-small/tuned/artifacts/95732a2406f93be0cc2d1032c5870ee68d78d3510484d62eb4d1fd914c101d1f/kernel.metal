#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device const half* parameter, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint tg_pos [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg_id [[simdgroup_index_in_threadgroup]], uint sg_width [[threads_per_simdgroup]], uint tg_threads [[threads_per_threadgroup]]) {
  if (tg_pos >= B) {
    return;
  }
  uint row = tg_pos;
  uint base = row * D;
  float v0 = 0.0f;
  float v1 = 0.0f;
  uint c0 = tid;
  uint c1 = tid + tg_threads;
  if (c0 < D) {
    v0 = float(x[base + c0]);
  }
  if (c1 < D) {
    v1 = float(x[base + c1]);
  }
  float sum = v0 * v0 + v1 * v1;
  for (uint col = tid + 2u * tg_threads; col < D; col += tg_threads) {
    float v = float(x[base + col]);
    sum += v * v;
  }
  float sg_sum = simd_sum(sum);
  threadgroup float partial[256];
  if (lane == 0u) {
    partial[sg_id] = sg_sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg_id == 0u) {
    uint sgs_per_tg = tg_threads / sg_width;
    float p = (lane < sgs_per_tg) ? partial[lane] : 0.0f;
    float total = simd_sum(p);
    if (lane == 0u) {
      partial[0] = rsqrt(total / float(D) + epsilon);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float scale = partial[0];
  if (c0 < D) {
    out[base + c0] = half(v0 * scale * float(parameter[c0]));
  }
  if (c1 < D) {
    out[base + c1] = half(v1 * scale * float(parameter[c1]));
  }
  for (uint col = tid + 2u * tg_threads; col < D; col += tg_threads) {
    float v = float(x[base + col]);
    out[base + col] = half(v * scale * float(parameter[col]));
  }
}
