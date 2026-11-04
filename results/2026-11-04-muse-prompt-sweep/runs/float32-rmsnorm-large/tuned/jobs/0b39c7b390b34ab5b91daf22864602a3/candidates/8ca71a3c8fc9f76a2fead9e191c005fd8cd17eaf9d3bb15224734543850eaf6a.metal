#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 threadgroup_pos [[threadgroup_position_in_grid]], uint3 thread_pos [[thread_position_in_threadgroup]], uint3 threads_per_tg [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint simdgroup_idx [[simdgroup_index_in_threadgroup]], uint threads_per_sg [[threads_per_simdgroup]]) {
  uint row = threadgroup_pos.x;
  if (row >= B) return;
  uint tid = thread_pos.x;
  uint nt = threads_per_tg.x;
  threadgroup float scratch[256];
  if (D == 4096u && nt == 256u) {
    uint base_vec = row * 1024u;
    float4 v0 = ((device const float4*)x)[base_vec + tid];
    float4 v1 = ((device const float4*)x)[base_vec + tid + 256u];
    float4 v2 = ((device const float4*)x)[base_vec + tid + 512u];
    float4 v3 = ((device const float4*)x)[base_vec + tid + 768u];
    float acc = dot(v0, v0) + dot(v1, v1) + dot(v2, v2) + dot(v3, v3);
    float sg_sum = simd_sum(acc);
    if (lane == 0u) scratch[simdgroup_idx] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
      uint num_sg = (nt + threads_per_sg - 1u) / threads_per_sg;
      float total = 0.0f;
      for (uint i = 0u; i < num_sg; ++i) total += scratch[i];
      scratch[0] = rsqrt(total / float(D) + epsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = scratch[0];
    float4 w0 = ((device const float4*)parameter)[tid];
    float4 w1 = ((device const float4*)parameter)[tid + 256u];
    float4 w2 = ((device const float4*)parameter)[tid + 512u];
    float4 w3 = ((device const float4*)parameter)[tid + 768u];
    ((device float4*)out)[base_vec + tid] = v0 * inv * w0;
    ((device float4*)out)[base_vec + tid + 256u] = v1 * inv * w1;
    ((device float4*)out)[base_vec + tid + 512u] = v2 * inv * w2;
    ((device float4*)out)[base_vec + tid + 768u] = v3 * inv * w3;
    return;
  }
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
  if (tid == 0u) {
    uint num_sg = (nt + threads_per_sg - 1u) / threads_per_sg;
    float total = 0.0f;
    for (uint i = 0u; i < num_sg; ++i) total += scratch[i];
    scratch[0] = rsqrt(total / float(D) + epsilon);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float inv = scratch[0];
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
