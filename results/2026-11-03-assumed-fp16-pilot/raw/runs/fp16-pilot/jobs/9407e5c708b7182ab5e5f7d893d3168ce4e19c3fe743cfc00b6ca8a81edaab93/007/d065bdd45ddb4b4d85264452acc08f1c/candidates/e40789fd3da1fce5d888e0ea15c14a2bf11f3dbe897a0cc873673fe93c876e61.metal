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
    if (D == 4096u && nthreads == 256u) {
        device const half4* x4 = reinterpret_cast<device const half4*>(x + base);
        device const half4* p4 = reinterpret_cast<device const half4*>(parameter);
        device half4* o4 = reinterpret_cast<device half4*>(out + base);
        float4 v0 = float4(x4[lid]);
        float4 v1 = float4(x4[lid + 256u]);
        float4 v2 = float4(x4[lid + 512u]);
        float4 v3 = float4(x4[lid + 768u]);
        float sum = dot(v0, v0) + dot(v1, v1) + dot(v2, v2) + dot(v3, v3);
        sdata[lid] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128u; stride > 0u; stride >>= 1u) {
            if (lid < stride) sdata[lid] += sdata[lid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lid == 0u) snorm = rsqrt(sdata[0] * (1.0f / 4096.0f) + epsilon);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float norm = snorm;
        o4[lid] = half4(v0 * norm * float4(p4[lid]));
        o4[lid + 256u] = half4(v1 * norm * float4(p4[lid + 256u]));
        o4[lid + 512u] = half4(v2 * norm * float4(p4[lid + 512u]));
        o4[lid + 768u] = half4(v3 * norm * float4(p4[lid + 768u]));
        return;
    }
    float sum = 0.0f;
    uint D4 = D / 4u;
    if (D4 > 0u) {
        device const half4* x4 = reinterpret_cast<device const half4*>(x + base);
        for (uint i = lid; i < D4; i += nthreads) {
            float4 v = float4(x4[i]);
            sum += dot(v, v);
        }
    }
    for (uint i = D4 * 4u + lid; i < D; i += nthreads) {
        float v = float(x[base + i]);
        sum += v * v;
    }
    sdata[lid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = nthreads >> 1u; stride > 0u; stride >>= 1u) {
        if (lid < stride && lid + stride < 256u) sdata[lid] += sdata[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) snorm = rsqrt(sdata[0] / float(D) + epsilon);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float norm = snorm;
    if (D4 > 0u) {
        device const half4* x4 = reinterpret_cast<device const half4*>(x + base);
        device const half4* p4 = reinterpret_cast<device const half4*>(parameter);
        device half4* o4 = reinterpret_cast<device half4*>(out + base);
        for (uint i = lid; i < D4; i += nthreads) {
            float4 v = float4(x4[i]);
            float4 p = float4(p4[i]);
            o4[i] = half4(v * norm * p);
        }
    }
    for (uint i = D4 * 4u + lid; i < D; i += nthreads) {
        float v = float(x[base + i]);
        float p = float(parameter[i]);
        out[base + i] = half(v * norm * p);
    }
}
