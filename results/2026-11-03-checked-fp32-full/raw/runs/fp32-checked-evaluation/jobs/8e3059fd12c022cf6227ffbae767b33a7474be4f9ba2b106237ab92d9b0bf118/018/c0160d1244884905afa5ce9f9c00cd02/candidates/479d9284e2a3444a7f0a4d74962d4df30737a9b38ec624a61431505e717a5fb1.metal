#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x [[buffer(0)]], device const float* parameter [[buffer(1)]], device float* out [[buffer(2)]], constant uint& B [[buffer(3)]], constant uint& D [[buffer(4)]], constant float& epsilon [[buffer(5)]], uint3 tg [[threadgroup_position_in_grid]], uint3 tp [[thread_position_in_threadgroup]], uint3 tptg [[threads_per_threadgroup]]) {
  threadgroup float buf[256];
  uint row = tg.x;
  uint tid = tp.x;
  uint n = tptg.x;
  if (row >= B) return;
  if (n > 256u) return;
  uint base = row * D;
  float sum = 0.0f;
  for (uint i = tid; i < D; i += n) {
    float v = x[base + i];
    sum += v * v;
  }
  buf[tid] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = n >> 1; stride > 0u; stride >>= 1) {
    if (tid < stride) buf[tid] += buf[tid + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float scale = rsqrt(buf[0] / float(D) + epsilon);
  for (uint i = tid; i < D; i += n) {
    out[base + i] = float(x[base + i] * scale * parameter[i]);
  }
}
