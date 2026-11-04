#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tg_id [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 tgs [[threads_per_threadgroup]]) {
  threadgroup float red[256];
  threadgroup float ecache[1024];
  (void)epsilon;
  uint row = tg_id.x;
  if (row >= B) { return; }
  uint lid = tid.x;
  uint width = tgs.x;
  if (width == 0u || lid >= 256u) { return; }
  device const half* rp = x + row * D;
  device half* op = out + row * D;
  float local_max = -3.4028235e38f;
  for (uint i = lid; i < D; i += width) { local_max = max(local_max, float(rp[i])); }
  red[lid] = local_max;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = width >> 1; stride > 0u; stride >>= 1) {
    if (lid < stride) { red[lid] = max(red[lid], red[lid + stride]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float row_max = red[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bool use_cache = D <= 1024u;
  float local_sum = 0.0f;
  for (uint i = lid; i < D; i += width) {
    float e = exp(float(rp[i]) - row_max);
    if (use_cache) { ecache[i] = e; }
    local_sum += e;
  }
  red[lid] = local_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = width >> 1; stride > 0u; stride >>= 1) {
    if (lid < stride) { red[lid] += red[lid + stride]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float inv_sum = 1.0f / red[0];
  for (uint i = lid; i < D; i += width) {
    float e = use_cache ? ecache[i] : exp(float(rp[i]) - row_max);
    op[i] = half(e * inv_sum);
  }
}
