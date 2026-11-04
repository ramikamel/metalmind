#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint row [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]], uint tg_size [[threads_per_threadgroup]]) {
    if (row >= B) return;
    threadgroup float partial[256];
    uint base = row * D;
    float sum = 0.0f;
    if ((D & 3u) == 0u) {
        device const float4* x4 = (device const float4*)(x + base);
        uint D4 = D >> 2;
        for (uint i = lid; i < D4; i += tg_size) {
            float4 v = x4[i];
            sum += dot(v, v);
        }
    } else {
        for (uint i = lid; i < D; i += tg_size) {
            float v = x[base + i];
            sum += v * v;
        }
    }
    partial[lid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size >> 1; stride > 0u; stride >>= 1) {
        if (lid < stride) partial[lid] += partial[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float scale = rsqrt(partial[0] / float(D) + epsilon);
    if ((D & 3u) == 0u) {
        device const float4* x4 = (device const float4*)(x + base);
        device const float4* p4 = (device const float4*)parameter;
        device float4* o4 = (device float4*)(out + base);
        uint D4 = D >> 2;
        float4 s = float4(scale);
        for (uint i = lid; i < D4; i += tg_size) {
            o4[i] = x4[i] * s * p4[i];
        }
    } else {
        for (uint i = lid; i < D; i += tg_size) {
            out[base + i] = x[base + i] * scale * parameter[i];
        }
    }
}
