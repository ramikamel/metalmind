#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint tid [[thread_index_in_threadgroup]], uint gid [[threadgroup_position_in_grid]], uint ngroups [[threadgroups_per_grid]], uint tgsize [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint simd_id [[simdgroup_index_in_threadgroup]], uint simd_size [[threads_per_simdgroup]]) {
  (void)epsilon;
  (void)tid;
  uint nsg = (tgsize + simd_size - 1u) / simd_size;
  uint row0 = gid * nsg + simd_id;
  uint rowStride = ngroups * nsg;
  for (uint row = row0; row < B; row += rowStride) {
    uint base = row * D;
    if (D == 1024u && simd_size == 32u) {
      device const float4* x4 = (device const float4*)(x + base);
      float4 v0 = x4[lane];
      float4 v1 = x4[lane + 32u];
      float4 v2 = x4[lane + 64u];
      float4 v3 = x4[lane + 96u];
      float4 v4 = x4[lane + 128u];
      float4 v5 = x4[lane + 160u];
      float4 v6 = x4[lane + 192u];
      float4 v7 = x4[lane + 224u];
      float lm = max(max(v0.x, v0.y), max(v0.z, v0.w));
      lm = max(lm, max(max(v1.x, v1.y), max(v1.z, v1.w)));
      lm = max(lm, max(max(v2.x, v2.y), max(v2.z, v2.w)));
      lm = max(lm, max(max(v3.x, v3.y), max(v3.z, v3.w)));
      lm = max(lm, max(max(v4.x, v4.y), max(v4.z, v4.w)));
      lm = max(lm, max(max(v5.x, v5.y), max(v5.z, v5.w)));
      lm = max(lm, max(max(v6.x, v6.y), max(v6.z, v6.w)));
      lm = max(lm, max(max(v7.x, v7.y), max(v7.z, v7.w)));
      float rm = simd_max(lm);
      float c = 1.4426950408889634f;
      float4 e0 = exp2((v0 - rm) * c);
      float4 e1 = exp2((v1 - rm) * c);
      float4 e2 = exp2((v2 - rm) * c);
      float4 e3 = exp2((v3 - rm) * c);
      float4 e4 = exp2((v4 - rm) * c);
      float4 e5 = exp2((v5 - rm) * c);
      float4 e6 = exp2((v6 - rm) * c);
      float4 e7 = exp2((v7 - rm) * c);
      float ls = ((e0.x + e0.y) + (e0.z + e0.w)) + ((e1.x + e1.y) + (e1.z + e1.w));
      ls += ((e2.x + e2.y) + (e2.z + e2.w)) + ((e3.x + e3.y) + (e3.z + e3.w));
      ls += ((e4.x + e4.y) + (e4.z + e4.w)) + ((e5.x + e5.y) + (e5.z + e5.w));
      ls += ((e6.x + e6.y) + (e6.z + e6.w)) + ((e7.x + e7.y) + (e7.z + e7.w));
      float inv = 1.0f / simd_sum(ls);
      device float4* o4 = (device float4*)(out + base);
      o4[lane] = e0 * inv;
      o4[lane + 32u] = e1 * inv;
      o4[lane + 64u] = e2 * inv;
      o4[lane + 96u] = e3 * inv;
      o4[lane + 128u] = e4 * inv;
      o4[lane + 160u] = e5 * inv;
      o4[lane + 192u] = e6 * inv;
      o4[lane + 224u] = e7 * inv;
    } else {
      float lm = -3.402823466e+38f;
      for (uint i = lane; i < D; i += simd_size) lm = max(lm, x[base + i]);
      float rm = simd_max(lm);
      float ls = 0.0f;
      for (uint i = lane; i < D; i += simd_size) ls += exp(x[base + i] - rm);
      float inv = 1.0f / simd_sum(ls);
      for (uint i = lane; i < D; i += simd_size) out[base + i] = exp(x[base + i] - rm) * inv;
    }
  }
}
