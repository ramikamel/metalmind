#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 gid [[thread_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 tpg [[threads_per_threadgroup]], uint3 tgpos [[threadgroup_position_in_grid]]) {
  (void)gid;
  (void)epsilon;
  uint row = tgpos.x;
  uint lid = tid.x;
  uint nthreads = tpg.x;
  threadgroup float shared[256];
  if (row < B) {
    uint base = row * D;
    float local_max = -3.402823466e+38f;
    for (uint i = lid; i < D; i += nthreads) {
      float v = x[base + i];
      local_max = max(local_max, v);
    }
    shared[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = nthreads / 2; stride > 0; stride >>= 1) {
      if (lid < stride) {
        shared[lid] = max(shared[lid], shared[lid + stride]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_sum = 0.0f;
    for (uint i = lid; i < D; i += nthreads) {
      local_sum += exp(x[base + i] - row_max);
    }
    shared[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = nthreads / 2; stride > 0; stride >>= 1) {
      if (lid < stride) {
        shared[lid] += shared[lid + stride];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_sum = shared[0];
    float inv_sum = 1.0f / row_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lid; i < D; i += nthreads) {
      float v = exp(x[base + i] - row_max) * inv_sum;
      out[base + i] = float(v);
    }
  }
}
