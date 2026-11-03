#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint2 gid [[thread_position_in_grid]], uint2 tid [[thread_position_in_threadgroup]], uint2 tgs [[threads_per_threadgroup]]) {
  uint row = gid.y;
  if (row >= B || D == 0) return;
  threadgroup float sdata[256];
  uint base = row * D;
  float sum = 0.0f;
  for (uint c = tid.x; c < D; c += tgs.x) {
    float v = x[base + c];
    sum += v * v;
  }
  sdata[tid.x] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = tgs.x / 2; stride > 0; stride >>= 1) {
    if (tid.x < stride) sdata[tid.x] += sdata[tid.x + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float scale = rsqrt(sdata[0] / float(D) + epsilon);
  for (uint c = tid.x; c < D; c += tgs.x) {
    float y = x[base + c] * scale * parameter[c];
    out[base + c] = y;
  }
}
