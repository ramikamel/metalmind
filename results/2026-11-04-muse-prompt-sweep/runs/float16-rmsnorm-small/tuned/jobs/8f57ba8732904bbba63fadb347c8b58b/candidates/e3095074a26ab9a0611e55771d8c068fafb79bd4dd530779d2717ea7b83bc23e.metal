#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device const half* parameter, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint tg_pos [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sg_id [[simdgroup_index_in_threadgroup]], uint sg_width [[threads_per_simdgroup]], uint tg_threads [[threads_per_threadgroup]], uint tg_count [[threadgroups_per_grid]]) {
  uint sgs_per_tg = tg_threads / sg_width;
  uint total_sgs = tg_count * sgs_per_tg;
  uint global_sg = tg_pos * sgs_per_tg + sg_id;
  for (uint row = global_sg; row < B; row += total_sgs) {
    uint base = row * D;
    float sum = 0.0f;
    for (uint col = lane; col < D; col += sg_width) {
      float v = float(x[base + col]);
      sum += v * v;
    }
    float total = simd_sum(sum);
    float scale = rsqrt(total / float(D) + epsilon);
    for (uint col = lane; col < D; col += sg_width) {
      float v = float(x[base + col]);
      float w = float(parameter[col]);
      out[base + col] = half(v * scale * w);
    }
  }
}
