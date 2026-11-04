#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tg_pos [[threadgroup_position_in_grid]], uint3 tid_in_tg [[thread_position_in_threadgroup]], uint3 threads_per_tg [[threads_per_threadgroup]]) {
  uint row = tg_pos.x;
  uint tid = tid_in_tg.x;
  uint tsize = threads_per_tg.x;
  if (row >= B) { return; }
  uint base = row * D;
  float local = 0.0f;
  for (uint i = tid; i < D; i += tsize) {
    float v = x[base + i];
    local += v * v;
  }
  threadgroup float shared[256];
  shared[tid] = local;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) { shared[tid] += shared[tid + stride]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float inv = rsqrt(shared[0] / float(D) + epsilon);
  for (uint i = tid; i < D; i += tsize) {
    float v = x[base + i];
    out[base + i] = float(v * inv * parameter[i]);
  }
}
