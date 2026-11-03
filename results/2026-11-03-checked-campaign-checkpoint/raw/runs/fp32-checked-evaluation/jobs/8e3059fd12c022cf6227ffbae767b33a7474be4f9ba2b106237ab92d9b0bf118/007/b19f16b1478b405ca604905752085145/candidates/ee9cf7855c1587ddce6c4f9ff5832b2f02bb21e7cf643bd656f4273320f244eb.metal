#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tg_size [[threads_per_threadgroup]]) {
    if (row >= B) return;
    uint base = row * D;
    uint d4 = D / 4u;
    device const float4* x4 = (device const float4*)x;
    device const float4* p4 = (device const float4*)parameter;
    device float4* o4 = (device float4*)out;
    float sum = 0.0f;
    for (uint i = tid; i < d4; i += tg_size) {
        float4 v = x4[row * d4 + i];
        sum += dot(v, v);
    }
    for (uint i = d4 * 4u + tid; i < D; i += tg_size) {
        float v = x[base + i];
        sum += v * v;
    }
    threadgroup float shared[256];
    shared[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float scale = rsqrt(shared[0] / float(D) + epsilon);
    for (uint i = tid; i < d4; i += tg_size) {
        float4 v = x4[row * d4 + i];
        o4[row * d4 + i] = v * scale * p4[i];
    }
    for (uint i = d4 * 4u + tid; i < D; i += tg_size) {
        out[base + i] = x[base + i] * scale * parameter[i];
    }
}
