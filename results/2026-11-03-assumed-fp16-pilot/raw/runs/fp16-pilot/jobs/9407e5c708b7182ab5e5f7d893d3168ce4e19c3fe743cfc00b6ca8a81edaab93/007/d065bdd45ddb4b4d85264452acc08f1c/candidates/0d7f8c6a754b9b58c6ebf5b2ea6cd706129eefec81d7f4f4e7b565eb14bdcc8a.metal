#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device const half* parameter, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 gid [[thread_position_in_grid]], uint3 tg_pos [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 tg_size [[threads_per_threadgroup]]) {
    uint row = tg_pos.x;
    if (row >= B) return;
    uint lid = tid.x;
    uint nthreads = tg_size.x;
    uint base = row * D;
    threadgroup float sdata[256];
    threadgroup float snorm;
    float sum = 0.0f;
    uint D4 = D / 4;
    if (D4 > 0) {
        device const half4* x4 = reinterpret_cast<device const half4*>(x + base);
        for (uint i = lid; i < D4; i += nthreads) {
            float4 v = float4(x4[i]);
            sum += dot(v, v);
        }
    }
    for (uint i = D4 * 4 + lid; i < D; i += nthreads) {
        float v = float(x[base + i]);
        sum += v * v;
    }
    sdata[lid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = nthreads >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) sdata[lid] += sdata[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) snorm = rsqrt(sdata[0] / float(D) + epsilon);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float norm = snorm;
    if (D4 > 0) {
        device const half4* x4 = reinterpret_cast<device const half4*>(x + base);
        device const half4* p4 = reinterpret_cast<device const half4*>(parameter);
        device half4* o4 = reinterpret_cast<device half4*>(out + base);
        for (uint i = lid; i < D4; i += nthreads) {
            float4 v = float4(x4[i]);
            float4 p = float4(p4[i]);
            o4[i] = half4(v * norm * p);
        }
    }
    for (uint i = D4 * 4 + lid; i < D; i += nthreads) {
        float v = float(x[base + i]);
        float p = float(parameter[i]);
        out[base + i] = half(v * norm * p);
    }
}
