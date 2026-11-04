#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tgid [[threadgroup_position_in_grid]], uint3 tpos [[thread_position_in_threadgroup]], uint3 tgsize [[threads_per_threadgroup]]) {
  uint row = tgid.x;
  uint tid = tpos.x;
  uint tgs = tgsize.x;
  threadgroup float partial[256];
  float sum = 0.0f;
  bool valid = row < B;
  if (valid) {
    uint base = row * D;
    if ((D & 3u) == 0u) {
      uint d4 = D >> 2u;
      device const float4* x4 = reinterpret_cast<device const float4*>(x + base);
      for (uint i = tid; i < d4; i += tgs) {
        float4 v = x4[i];
        sum += dot(v, v);
      }
    } else {
      for (uint i = tid; i < D; i += tgs) {
        float v = x[base + i];
        sum += v * v;
      }
    }
  }
  if (tid < 256u) partial[tid] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = tgs >> 1u; stride > 0u; stride >>= 1u) {
    if (tid < stride) partial[tid] += partial[tid + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float inv = rsqrt(partial[0] / float(D) + epsilon);
  if (valid) {
    uint base = row * D;
    if ((D & 3u) == 0u) {
      uint d4 = D >> 2u;
      device const float4* x4 = reinterpret_cast<device const float4*>(x + base);
      device const float4* p4 = reinterpret_cast<device const float4*>(parameter);
      device float4* o4 = reinterpret_cast<device float4*>(out + base);
      for (uint i = tid; i < d4; i += tgs) {
        float4 v = x4[i];
        float4 w = p4[i];
        o4[i] = v * inv * w;
      }
    } else {
      for (uint i = tid; i < D; i += tgs) {
        out[base + i] = x[base + i] * inv * parameter[i];
      }
    }
  }
}
