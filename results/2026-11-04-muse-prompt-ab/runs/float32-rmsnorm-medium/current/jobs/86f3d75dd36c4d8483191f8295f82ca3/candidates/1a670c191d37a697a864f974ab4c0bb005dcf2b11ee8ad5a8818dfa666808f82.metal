#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tg [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]], uint3 tgsize [[threads_per_threadgroup]]) {
    uint row = tg.x;
    if (row >= B) { return; }
    uint lane = lid.x;
    uint nt = tgsize.x;
    threadgroup float partial[256];
    if (((D & 3u) == 0u) && ((D >> 2u) <= nt) && (nt <= 256u)) {
        uint d4 = D >> 2u;
        bool active = lane < d4;
        device const float4* x4 = reinterpret_cast<device const float4*>(x);
        device const float4* p4 = reinterpret_cast<device const float4*>(parameter);
        device float4* o4 = reinterpret_cast<device float4*>(out);
        float4 v = float4(0.0f);
        if (active) { v = x4[row * d4 + lane]; }
        float s = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        partial[lane] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = nt >> 1u; stride > 0u; stride >>= 1u) {
            if (lane < stride) { partial[lane] += partial[lane + stride]; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float inv = rsqrt(partial[0] / float(D) + epsilon);
        if (active) { o4[row * d4 + lane] = v * inv * p4[lane]; }
        return;
    }
    uint base = row * D;
    float s = 0.0f;
    for (uint i = lane; i < D; i += nt) { float v = x[base + i]; s += v * v; }
    if (lane < 256u) { partial[lane] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = nt >> 1u; stride > 0u; stride >>= 1u) {
        if (lane < stride && (lane + stride) < 256u) { partial[lane] += partial[lane + stride]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(partial[0] / float(D) + epsilon);
    for (uint i = lane; i < D; i += nt) { out[base + i] = x[base + i] * inv * parameter[i]; }
}
