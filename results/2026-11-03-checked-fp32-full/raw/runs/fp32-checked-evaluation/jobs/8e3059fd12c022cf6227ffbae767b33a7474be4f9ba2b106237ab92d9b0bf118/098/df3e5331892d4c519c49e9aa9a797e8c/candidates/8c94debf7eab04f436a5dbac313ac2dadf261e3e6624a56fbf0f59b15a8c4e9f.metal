#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tg_pos [[threadgroup_position_in_grid]], uint3 tid_in_tg [[thread_position_in_threadgroup]], uint3 threads_per_tg [[threads_per_threadgroup]]) {
    uint row = tg_pos.x;
    if (row >= B) return;
    uint tid = tid_in_tg.x;
    uint tsz = threads_per_tg.x;
    uint base = row * D;
    float sum = 0.0f;
    bool vec = ((D & 3u) == 0u);
    if (vec) {
        uint D4 = D >> 2u;
        device const float4* x4 = reinterpret_cast<device const float4*>(x + base);
        for (uint i = tid; i < D4; i += tsz) {
            float4 v = x4[i];
            sum += dot(v, v);
        }
    } else {
        for (uint i = tid; i < D; i += tsz) {
            float v = x[base + i];
            sum += v * v;
        }
    }
    threadgroup float partial[256];
    partial[tid] = sum;
    threadgroup_barrier(mem_threadgroup);
    for (uint stride = tsz >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_threadgroup);
    }
    float inv = rsqrt(partial[0] / float(D) + epsilon);
    if (vec) {
        uint D4 = D >> 2u;
        device const float4* x4 = reinterpret_cast<device const float4*>(x + base);
        device const float4* p4 = reinterpret_cast<device const float4*>(parameter);
        device float4* o4 = reinterpret_cast<device float4*>(out + base);
        for (uint i = tid; i < D4; i += tsz) {
            float4 v = x4[i];
            o4[i] = v * inv * p4[i];
        }
    } else {
        for (uint i = tid; i < D; i += tsz) {
            out[base + i] = x[base + i] * inv * parameter[i];
        }
    }
}
