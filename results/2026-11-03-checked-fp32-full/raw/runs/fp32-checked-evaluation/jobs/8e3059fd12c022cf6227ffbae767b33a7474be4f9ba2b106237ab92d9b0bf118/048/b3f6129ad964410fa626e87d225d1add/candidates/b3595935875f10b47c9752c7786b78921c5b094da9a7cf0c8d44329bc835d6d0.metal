#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]], uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]], uint3 threads_per_threadgroup [[threads_per_threadgroup]]) {
    uint row = threadgroup_position_in_grid.x;
    if (row >= B) return;
    uint tid = thread_position_in_threadgroup.x;
    uint tgs = threads_per_threadgroup.x;
    threadgroup float partial[256];
    uint base = row * D;
    uint n4 = D >> 2;
    device const float4* x4 = (device const float4*)x;
    device const float4* p4 = (device const float4*)parameter;
    device float4* o4 = (device float4*)out;
    uint base4 = row * n4;
    float sum = 0.0f;
    for (uint i = tid; i < n4; i += tgs) {
        float4 v = x4[base4 + i];
        sum += dot(v, v);
    }
    uint rem = D & 3u;
    if (rem != 0u && tid < rem) {
        float v = x[base + n4 * 4u + tid];
        sum += v * v;
    }
    if (tid < 256u) partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgs >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(partial[0] / float(D) + epsilon);
    for (uint i = tid; i < n4; i += tgs) {
        float4 v = x4[base4 + i];
        float4 p = p4[i];
        o4[base4 + i] = v * inv * p;
    }
    if (rem != 0u && tid < rem) {
        uint idx = n4 * 4u + tid;
        out[base + idx] = x[base + idx] * inv * parameter[idx];
    }
}
