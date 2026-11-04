#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint tid [[thread_position_in_threadgroup]], uint tgid [[threadgroup_position_in_grid]], uint tgdim [[threads_per_threadgroup]]) {
  if (tgid >= B) return;
  threadgroup float partials[256];
  uint D4 = D / 4u;
  device const float4* x4 = (device const float4*)x;
  device const float4* p4 = (device const float4*)parameter;
  device float4* o4 = (device float4*)out;
  uint row4 = tgid * D4;
  float sum = 0.0f;
  for (uint i = tid; i < D4; i += tgdim) {
    float4 v = x4[row4 + i];
    sum += dot(v, v);
  }
  for (uint i = D4 * 4u + tid; i < D; i += tgdim) {
    float v = x[(uint)tgid * D + i];
    sum += v * v;
  }
  if (tid < 256u) partials[tid] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128u; stride > 0u; stride >>= 1) {
    if (tid < stride) partials[tid] += partials[tid + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float scale = rsqrt(partials[0] / float(D) + epsilon);
  for (uint i = tid; i < D4; i += tgdim) {
    float4 v = x4[row4 + i];
    o4[row4 + i] = v * scale * p4[i];
  }
  for (uint i = D4 * 4u + tid; i < D; i += tgdim) {
    uint idx = (uint)tgid * D + i;
    out[idx] = x[idx] * scale * parameter[i];
  }
}
