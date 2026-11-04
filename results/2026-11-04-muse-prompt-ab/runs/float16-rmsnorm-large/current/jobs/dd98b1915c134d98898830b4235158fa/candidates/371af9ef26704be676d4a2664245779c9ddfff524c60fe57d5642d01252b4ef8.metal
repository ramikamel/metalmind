#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device const half* parameter, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tg_pos [[threadgroup_position_in_grid]], uint3 tid3 [[thread_position_in_threadgroup]], uint3 tg_dim [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sgrp [[simd_group_id]], uint nsg [[simdgroups_per_threadgroup]]) {
    uint row = tg_pos.x;
    uint tid = tid3.x;
    uint tg_size = tg_dim.x;
    if (row >= B) {
        return;
    }
    uint base = row * D;
    threadgroup float partial[256];
    threadgroup float inv_shared;
    if (D == 4096u && tg_size == 256u) {
        device const half4* x4 = reinterpret_cast<device const half4*>(x + base);
        device const half4* p4 = reinterpret_cast<device const half4*>(parameter);
        device half4* o4 = reinterpret_cast<device half4*>(out + base);
        half4 v0 = x4[tid];
        half4 v1 = x4[tid + 256u];
        half4 v2 = x4[tid + 512u];
        half4 v3 = x4[tid + 768u];
        float sum = 0.0f;
        sum += float(v0.x) * float(v0.x) + float(v0.y) * float(v0.y) + float(v0.z) * float(v0.z) + float(v0.w) * float(v0.w);
        sum += float(v1.x) * float(v1.x) + float(v1.y) * float(v1.y) + float(v1.z) * float(v1.z) + float(v1.w) * float(v1.w);
        sum += float(v2.x) * float(v2.x) + float(v2.y) * float(v2.y) + float(v2.z) * float(v2.z) + float(v2.w) * float(v2.w);
        sum += float(v3.x) * float(v3.x) + float(v3.y) * float(v3.y) + float(v3.z) * float(v3.z) + float(v3.w) * float(v3.w);
        float s = simd_sum(sum);
        if (lane == 0u) {
            partial[sgrp] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgrp == 0u) {
            float v = (lane < nsg) ? partial[lane] : 0.0f;
            v = simd_sum(v);
            if (lane == 0u) {
                inv_shared = rsqrt(v * (1.0f / 4096.0f) + epsilon);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv = inv_shared;
        half4 p0 = p4[tid];
        half4 p1 = p4[tid + 256u];
        half4 p2 = p4[tid + 512u];
        half4 p3 = p4[tid + 768u];
        o4[tid] = half4(half(float(v0.x) * inv * float(p0.x)), half(float(v0.y) * inv * float(p0.y)), half(float(v0.z) * inv * float(p0.z)), half(float(v0.w) * inv * float(p0.w)));
        o4[tid + 256u] = half4(half(float(v1.x) * inv * float(p1.x)), half(float(v1.y) * inv * float(p1.y)), half(float(v1.z) * inv * float(p1.z)), half(float(v1.w) * inv * float(p1.w)));
        o4[tid + 512u] = half4(half(float(v2.x) * inv * float(p2.x)), half(float(v2.y) * inv * float(p2.y)), half(float(v2.z) * inv * float(p2.z)), half(float(v2.w) * inv * float(p2.w)));
        o4[tid + 768u] = half4(half(float(v3.x) * inv * float(p3.x)), half(float(v3.y) * inv * float(p3.y)), half(float(v3.z) * inv * float(p3.z)), half(float(v3.w) * inv * float(p3.w)));
        return;
    }
    float sum = 0.0f;
    if ((D & 3u) == 0u) {
        device const half4* x4 = reinterpret_cast<device const half4*>(x + base);
        uint d4 = D >> 2u;
        for (uint i = tid; i < d4; i += tg_size) {
            half4 v = x4[i];
            float a = float(v.x);
            float b = float(v.y);
            float c = float(v.z);
            float d = float(v.w);
            sum += a * a + b * b + c * c + d * d;
        }
    } else {
        for (uint i = tid; i < D; i += tg_size) {
            float v = float(x[base + i]);
            sum += v * v;
        }
    }
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        inv_shared = rsqrt(partial[0] / float(D) + epsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = inv_shared;
    if ((D & 3u) == 0u) {
        device const half4* x4 = reinterpret_cast<device const half4*>(x + base);
        device const half4* p4 = reinterpret_cast<device const half4*>(parameter);
        device half4* o4 = reinterpret_cast<device half4*>(out + base);
        uint d4 = D >> 2u;
        for (uint i = tid; i < d4; i += tg_size) {
            half4 v = x4[i];
            half4 p = p4[i];
            o4[i] = half4(half(float(v.x) * inv * float(p.x)), half(float(v.y) * inv * float(p.y)), half(float(v.z) * inv * float(p.z)), half(float(v.w) * inv * float(p.w)));
        }
    } else {
        for (uint i = tid; i < D; i += tg_size) {
            float v = float(x[base + i]);
            float p = float(parameter[i]);
            out[base + i] = half(v * inv * p);
        }
    }
}
