#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tg_pos [[threadgroup_position_in_grid]], uint3 ti [[thread_position_in_threadgroup]], uint3 tpg [[threads_per_threadgroup]]) {
    uint row = tg_pos.x;
    uint tid = ti.x;
    uint tg = tpg.x;
    if (row >= B) return;
    device const float* row_x = x + row * D;
    device float* row_o = out + row * D;
    float sum = 0.0f;
    if ((D & 3u) == 0u) {
        device const float4* x4 = (device const float4*)row_x;
        uint d4 = D >> 2;
        for (uint i = tid; i < d4; i += tg) {
            float4 v = x4[i];
            sum += dot(v, v);
        }
    } else {
        for (uint i = tid; i < D; i += tg) {
            float v = row_x[i];
            sum += v * v;
        }
    }
    threadgroup float partial[256];
    if (tid < 256u) partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) partial[tid] += partial[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(partial[0] / float(D) + epsilon);
    if ((D & 3u) == 0u) {
        device const float4* x4 = (device const float4*)row_x;
        device const float4* p4 = (device const float4*)parameter;
        device float4* o4 = (device float4*)row_o;
        uint d4 = D >> 2;
        for (uint i = tid; i < d4; i += tg) {
            float4 v = x4[i];
            float4 p = p4[i];
            o4[i] = v * inv * p;
        }
    } else {
        for (uint i = tid; i < D; i += tg) {
            row_o[i] = row_x[i] * inv * parameter[i];
        }
    }
}
