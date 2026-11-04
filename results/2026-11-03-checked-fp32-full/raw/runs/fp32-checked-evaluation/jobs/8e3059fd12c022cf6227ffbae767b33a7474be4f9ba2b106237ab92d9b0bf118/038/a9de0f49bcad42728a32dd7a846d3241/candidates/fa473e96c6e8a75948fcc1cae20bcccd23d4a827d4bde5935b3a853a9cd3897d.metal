#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tg [[threadgroup_position_in_grid]], uint3 tid3 [[thread_position_in_threadgroup]], uint3 tpg [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint sgw [[threads_per_simdgroup]]) {
  uint row = tg.x;
  if (row >= B) { return; }
  uint tid = tid3.x;
  uint nthreads = tpg.x;
  device const float* xr = x + (uint)row * D;
  device float* outr = out + (uint)row * D;
  threadgroup float red[256];
  threadgroup float4 cache[1024];
  uint nsg = (nthreads + sgw - 1) / sgw;
  bool use4 = ((D & 3u) == 0u) && (D <= 4096u);
  if (use4) {
    uint D4 = D >> 2;
    device const float4* x4 = (device const float4*)xr;
    device float4* o4 = (device float4*)outr;
    float lmax = -INFINITY;
    for (uint i = tid; i < D4; i += nthreads) { float4 v = x4[i]; lmax = max(lmax, max(max(v.x, v.y), max(v.z, v.w))); }
    float smax = simd_max(lmax);
    if (lane == 0) { red[sg] = smax; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) { float v = (lane < nsg) ? red[lane] : -INFINITY; v = simd_max(v); if (lane == 0) { red[0] = v; } }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = red[0];
    float lsum = 0.0f;
    for (uint i = tid; i < D4; i += nthreads) { float4 e = exp(x4[i] - row_max); cache[i] = e; lsum += e.x + e.y + e.z + e.w; }
    float ssum = simd_sum(lsum);
    if (lane == 0) { red[sg] = ssum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) { float v = (lane < nsg) ? red[lane] : 0.0f; v = simd_sum(v); if (lane == 0) { red[0] = v; } }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = 1.0f / red[0];
    for (uint i = tid; i < D4; i += nthreads) { o4[i] = cache[i] * inv; }
  } else {
    float lmax = -INFINITY;
    for (uint i = tid; i < D; i += nthreads) { lmax = max(lmax, xr[i]); }
    float smax = simd_max(lmax);
    if (lane == 0) { red[sg] = smax; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) { float v = (lane < nsg) ? red[lane] : -INFINITY; v = simd_max(v); if (lane == 0) { red[0] = v; } }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = red[0];
    float lsum = 0.0f;
    for (uint i = tid; i < D; i += nthreads) { lsum += exp(xr[i] - row_max); }
    float ssum = simd_sum(lsum);
    if (lane == 0) { red[sg] = ssum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) { float v = (lane < nsg) ? red[lane] : 0.0f; v = simd_sum(v); if (lane == 0) { red[0] = v; } }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = 1.0f / red[0];
    for (uint i = tid; i < D; i += nthreads) { outr[i] = exp(xr[i] - row_max) * inv; }
  }
}
