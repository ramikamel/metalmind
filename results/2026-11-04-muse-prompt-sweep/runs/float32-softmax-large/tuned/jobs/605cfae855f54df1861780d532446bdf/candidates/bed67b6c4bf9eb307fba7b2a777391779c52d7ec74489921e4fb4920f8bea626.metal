#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tg [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 tptg [[threads_per_threadgroup]], uint lid [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  (void)epsilon;
  uint row = tg.x;
  if (row >= B) return;
  uint t = tid.x;
  uint nt = tptg.x;
  uint rowOff = row * D;
  threadgroup float shMax[32];
  threadgroup float shSum[32];
  if (t < 32u) { shMax[t] = -3.4028235e38f; shSum[t] = 0.0f; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float4 vals[4];
  float localMax = -3.4028235e38f;
  for (uint k = 0u; k < 4u; ++k) {
    uint base = (k * nt + t) * 4u;
    float4 v = float4(0.0f);
    if (base + 3u < D) {
      v = ((device const float4*)(x + rowOff + base))[0];
    } else if (base < D) {
      v.x = (base < D) ? x[rowOff + base] : -3.4028235e38f;
      v.y = (base + 1u < D) ? x[rowOff + base + 1u] : -3.4028235e38f;
      v.z = (base + 2u < D) ? x[rowOff + base + 2u] : -3.4028235e38f;
      v.w = (base + 3u < D) ? x[rowOff + base + 3u] : -3.4028235e38f;
      if (base >= D) v.x = -3.4028235e38f;
    } else {
      v = float4(-3.4028235e38f);
    }
    vals[k] = v;
    localMax = max(localMax, max(max(v.x, v.y), max(v.z, v.w)));
  }
  float sgMax = simd_max(localMax);
  if (lid == 0u) shMax[sg] = sgMax;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0u) {
    float m = shMax[0];
    for (uint i = 1u; i < 32u; ++i) m = max(m, shMax[i]);
    shMax[0] = m;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float rowMax = shMax[0];
  float localSum = 0.0f;
  for (uint k = 0u; k < 4u; ++k) {
    float4 e = exp(vals[k] - rowMax);
    vals[k] = e;
    localSum += (e.x + e.y) + (e.z + e.w);
  }
  float sgSum = simd_sum(localSum);
  if (lid == 0u) shSum[sg] = sgSum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0u) {
    float s = 0.0f;
    for (uint i = 0u; i < 32u; ++i) s += shSum[i];
    shSum[0] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float invSum = 1.0f / shSum[0];
  for (uint k = 0u; k < 4u; ++k) {
    uint base = (k * nt + t) * 4u;
    float4 o = vals[k] * invSum;
    if (base + 3u < D) {
      ((device float4*)(out + rowOff + base))[0] = o;
    } else {
      if (base < D) out[rowOff + base] = o.x;
      if (base + 1u < D) out[rowOff + base + 1u] = o.y;
      if (base + 2u < D) out[rowOff + base + 2u] = o.z;
      if (base + 3u < D) out[rowOff + base + 3u] = o.w;
    }
  }
}
