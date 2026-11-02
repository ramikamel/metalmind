#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint tid [[thread_index_in_threadgroup]], uint row [[threadgroup_position_in_grid]]) {
    if (row >= B) {
        return;
    }
    threadgroup float partial[256];
    if (D == 1024u) {
        float4 v = ((device const float4*)(x + row * 1024u))[tid];
        partial[tid] = dot(v, v);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint offset = 128u; offset > 0u; offset >>= 1u) {
            if (tid < offset) {
                partial[tid] += partial[tid + offset];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float inv = rsqrt(partial[0] * (1.0f / 1024.0f) + epsilon);
        float4 p = ((device const float4*)parameter)[tid];
        ((device float4*)out)[row * 256u + tid] = v * inv * p;
        return;
    }
    float sum = 0.0f;
    for (uint j = tid; j < D; j += 256u) {
        float v = x[row * D + j];
        sum += v * v;
    }
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 128u; offset > 0u; offset >>= 1u) {
        if (tid < offset) {
            partial[tid] += partial[tid + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(partial[0] / float(D) + epsilon);
    for (uint j = tid; j < D; j += 256u) {
        out[row * D + j] = x[row * D + j] * inv * parameter[j];
    }
}
