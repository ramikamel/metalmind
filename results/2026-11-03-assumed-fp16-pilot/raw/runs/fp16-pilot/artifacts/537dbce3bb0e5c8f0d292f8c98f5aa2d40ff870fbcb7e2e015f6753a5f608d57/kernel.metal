#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device const half* parameter, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tg [[threadgroup_position_in_grid]], uint3 tid_in_tg [[thread_position_in_threadgroup]], uint3 tgs [[threads_per_threadgroup]]) {
  uint row = tg.x;
  if (row >= B) {
    return;
  }
  uint tid = tid_in_tg.x;
  uint nt = tgs.x;
  threadgroup float sdata[256];
  float sum = 0.0f;
  uint base = row * D;
  for (uint j = tid; j < D; j += nt) {
    float v = float(x[base + j]);
    sum += v * v;
  }
  if (tid < 256) {
    sdata[tid] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint offset = 128; offset > 0; offset >>= 1) {
    if (tid < offset) {
      sdata[tid] += sdata[tid + offset];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float mean = sdata[0] / float(D);
  float inv = rsqrt(mean + epsilon);
  for (uint j = tid; j < D; j += nt) {
    float v = float(x[base + j]);
    float w = float(parameter[j]);
    out[base + j] = half(v * inv * w);
  }
}
