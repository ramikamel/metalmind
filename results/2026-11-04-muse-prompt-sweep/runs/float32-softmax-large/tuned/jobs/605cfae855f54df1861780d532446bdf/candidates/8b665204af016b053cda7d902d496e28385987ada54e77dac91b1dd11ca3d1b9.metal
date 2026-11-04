#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tg [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 tptg [[threads_per_threadgroup]], uint lid [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint w [[threads_per_simdgroup]]) {
  (void)epsilon;
  uint row = tg.x;
  if (row >= B) return;
  uint t = tid.x;
  uint nt = tptg.x;
  uint rowOff = row * D;
  threadgroup float shMax[32];
  threadgroup float shSum[32];
  float4 vals[8];
  float localMax = -3.4028235e38f;
  for (uint k = 0u; k < 8u; ++k) {
    uint base = (k * nt + t) * 4u;
    float4 v = float4(-3.4028235e38f);
    if (base + 3u < D) {
      v = ((device const float4*)(x + rowOff + base))[0];
    } else if (base < D) {
      v.x = x[rowOff + base];
      if (base + 1u < D) v.y = x[rowOff + base + 1u];
      if (base + 2u < D) v.z = x[rowOff + base + 2u];
      if (base + 3u < D) v.w = x[rowOff + base + 3u];
    }
    vals[k] = v;
    localMax = max(localMax, max(max(v.x, v.y), max(v.z, v.w)));
  }
  bool empty = (localMax == -3.4028235e38f);
  float localSum = 0.0f;
  for (uint k = 0u; k < 8u; ++k) {
    float4 e = empty ? float4(0.0f) : exp(vals[k] - localMax);
    vals[k] = e;
    localSum += (e.x + e.y) + (e.z + e.w);
  }
  float sgMax = simd_max(localMax);
  float scaledSum = empty ? 0.0f : localSum * exp(localMax - sgMax);
  float sgSum = simd_sum(scaledSum);
  if (lid == 0u) { shMax[sg] = sgMax; shSum[sg] = sgSum; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint numSg = (nt + w - 1u) / w;
  if (numSg > 32u) numSg = 32u;
  if (t == 0u) {
    float m = shMax[0];
    for (uint i = 1u; i < numSg; ++i) m = max(m, shMax[i]);
    float s = 0.0f;
    for (uint i = 0u; i < numSg; ++i) s += shSum[i] * exp(shMax[i] - m);
    shMax[0] = m;
    shSum[0] = 1.0f / s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float scale = empty ? 0.0f : exp(localMax - shMax[0]) * shSum[0];
  for (uint k = 0u; k < 8u; ++k) {
    uint base = (k * nt + t) * 4u;
    float4 o = vals[k] * scale;
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
