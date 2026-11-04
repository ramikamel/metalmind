#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 gid [[thread_position_in_grid]], uint3 tg [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]], uint3 tgs [[threads_per_threadgroup]]) {
    uint row = tg.x;
    if (row >= B) {
        return;
    }
    uint base = row * D;
    threadgroup float partial[256];
    bool fast = ((D & 3u) == 0u) && ((D >> 2u) == tgs.x) && (tgs.x <= 256u);
    if (fast) {
        device const float4* x4 = reinterpret_cast<device const float4*>(x + base);
        device const float4* p4 = reinterpret_cast<device const float4*>(parameter);
        device float4* o4 = reinterpret_cast<device float4*>(out + base);
        float4 v = x4[lid.x];
        partial[lid.x] = dot(v, v);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = tgs.x >> 1u; stride > 0u; stride >>= 1u) {
            if (lid.x < stride) {
                partial[lid.x] += partial[lid.x + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float inv = rsqrt(partial[0] / float(D) + epsilon);
        float4 p = p4[lid.x];
        o4[lid.x] = v * inv * p;
        return;
    }
    float sum = 0.0f;
    for (uint c = lid.x; c < D; c += tgs.x) {
        float v = x[base + c];
        sum += v * v;
    }
    if (lid.x < 256u) {
        partial[lid.x] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgs.x >> 1u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            partial[lid.x] += partial[lid.x + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(partial[0] / float(D) + epsilon);
    for (uint c = lid.x; c < D; c += tgs.x) {
        out[base + c] = x[base + c] * inv * parameter[c];
    }
}
