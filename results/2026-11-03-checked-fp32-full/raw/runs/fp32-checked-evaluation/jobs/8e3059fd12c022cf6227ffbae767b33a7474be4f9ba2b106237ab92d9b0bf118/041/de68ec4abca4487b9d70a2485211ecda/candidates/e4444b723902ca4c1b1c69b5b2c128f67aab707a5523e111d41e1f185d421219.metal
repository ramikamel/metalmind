#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 gid [[thread_position_in_grid]], uint3 tid3 [[thread_position_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]], uint3 tpg [[threads_per_threadgroup]]) {
    threadgroup float partial[256];
    threadgroup float4 tile[1024];
    uint tid = tid3.x;
    uint row = tg.x;
    if (row >= B) { return; }
    uint base = row * D;
    float local = 0.0f;
    bool vec = ((D & 3u) == 0u) && (D <= 4096u);
    if (vec) {
        uint D4 = D >> 2;
        device const float4* x4 = (device const float4*)(x + base);
        for (uint i = tid; i < D4; i += 256u) {
            float4 v = x4[i];
            tile[i] = v;
            local += dot(v, v);
        }
    } else {
        for (uint i = tid; i < D; i += 256u) {
            float v = x[base + i];
            local += v * v;
        }
    }
    partial[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128u; s > 0u; s >>= 1) {
        if (tid < s) { partial[tid] += partial[tid + s]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(partial[0] / float(D) + epsilon);
    if (vec) {
        uint D4 = D >> 2;
        device const float4* p4 = (device const float4*)parameter;
        device float4* o4 = (device float4*)(out + base);
        float4 scale = float4(inv);
        for (uint i = tid; i < D4; i += 256u) {
            float4 v = tile[i];
            float4 g = p4[i];
            o4[i] = v * scale * g;
        }
    } else {
        for (uint i = tid; i < D; i += 256u) {
            out[base + i] = x[base + i] * inv * parameter[i];
        }
    }
}
