#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 threadgroup_pos [[threadgroup_position_in_grid]], uint3 thread_pos [[thread_position_in_threadgroup]], uint3 threads_per_tg [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint simdgroup_idx [[simdgroup_index_in_threadgroup]], uint threads_per_sg [[threads_per_simdgroup]]) {
  uint tg = threadgroup_pos.x;
  uint tid = thread_pos.x;
  uint nt = threads_per_tg.x;
  threadgroup float scratch[256];
  if (D == 4096u && nt == 256u) {
    uint row0 = tg * 2u;
    if (row0 >= B) return;
    uint row1 = row0 + 1u;
    bool has1 = row1 < B;
    uint base0 = row0 * 1024u;
    uint base1 = has1 ? row1 * 1024u : base0;
    float4 v0 = ((device const float4*)x)[base0 + tid];
    float4 v1 = ((device const float4*)x)[base0 + tid + 256u];
    float4 v2 = ((device const float4*)x)[base0 + tid + 512u];
    float4 v3 = ((device const float4*)x)[base0 + tid + 768u];
    float4 u0 = float4(0.0f);
    float4 u1 = float4(0.0f);
    float4 u2 = float4(0.0f);
    float4 u3 = float4(0.0f);
    if (has1) {
      u0 = ((device const float4*)x)[base1 + tid];
      u1 = ((device const float4*)x)[base1 + tid + 256u];
      u2 = ((device const float4*)x)[base1 + tid + 512u];
      u3 = ((device const float4*)x)[base1 + tid + 768u];
    }
    float acc0 = dot(v0, v0) + dot(v1, v1) + dot(v2, v2) + dot(v3, v3);
    float acc1 = dot(u0, u0) + dot(u1, u1) + dot(u2, u2) + dot(u3, u3);
    float s0 = simd_sum(acc0);
    float s1 = simd_sum(acc1);
    if (lane == 0u) {
      scratch[simdgroup_idx] = s0;
      scratch[128u + simdgroup_idx] = s1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint num_sg = (nt + threads_per_sg - 1u) / threads_per_sg;
    float total0 = 0.0f;
    float total1 = 0.0f;
    for (uint i = 0u; i < num_sg; ++i) {
      total0 += scratch[i];
      total1 += scratch[128u + i];
    }
    float inv0 = rsqrt(total0 / float(D) + epsilon);
    float inv1 = rsqrt(total1 / float(D) + epsilon);
    float4 w0 = ((device const float4*)parameter)[tid];
    ((device float4*)out)[base0 + tid] = v0 * inv0 * w0;
    if (has1) ((device float4*)out)[base1 + tid] = u0 * inv1 * w0;
    float4 w1 = ((device const float4*)parameter)[tid + 256u];
    ((device float4*)out)[base0 + tid + 256u] = v1 * inv0 * w1;
    if (has1) ((device float4*)out)[base1 + tid + 256u] = u1 * inv1 * w1;
    float4 w2 = ((device const float4*)parameter)[tid + 512u];
    ((device float4*)out)[base0 + tid + 512u] = v2 * inv0 * w2;
    if (has1) ((device float4*)out)[base1 + tid + 512u] = u2 * inv1 * w2;
    float4 w3 = ((device const float4*)parameter)[tid + 768u];
    ((device float4*)out)[base0 + tid + 768u] = v3 * inv0 * w3;
    if (has1) ((device float4*)out)[base1 + tid + 768u] = u3 * inv1 * w3;
    return;
  }
  uint row = tg;
  if (row >= B) return;
  float acc = 0.0f;
  uint base = row * D;
  if ((D & 3u) == 0u) {
    uint vec_d = D >> 2;
    uint base_vec = row * vec_d;
    for (uint i = tid; i < vec_d; i += nt) {
      float4 v = ((device const float4*)x)[base_vec + i];
      acc += dot(v, v);
    }
  } else {
    for (uint i = tid; i < D; i += nt) {
      float v = x[base + i];
      acc += v * v;
    }
  }
  float sg_sum = simd_sum(acc);
  if (lane == 0u) scratch[simdgroup_idx] = sg_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint num_sg = (nt + threads_per_sg - 1u) / threads_per_sg;
  float total = 0.0f;
  for (uint i = 0u; i < num_sg; ++i) total += scratch[i];
  float inv = rsqrt(total / float(D) + epsilon);
  if ((D & 3u) == 0u) {
    uint vec_d = D >> 2;
    uint base_vec = row * vec_d;
    for (uint i = tid; i < vec_d; i += nt) {
      float4 v = ((device const float4*)x)[base_vec + i];
      float4 w = ((device const float4*)parameter)[i];
      ((device float4*)out)[base_vec + i] = v * inv * w;
    }
  } else {
    for (uint i = tid; i < D; i += nt) {
      out[base + i] = x[base + i] * inv * parameter[i];
    }
  }
}
