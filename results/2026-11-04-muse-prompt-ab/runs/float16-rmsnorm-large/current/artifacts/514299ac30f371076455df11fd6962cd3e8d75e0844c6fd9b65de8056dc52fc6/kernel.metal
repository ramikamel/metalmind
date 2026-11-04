#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device const half* parameter, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tg_pos [[threadgroup_position_in_grid]], uint3 tid3 [[thread_position_in_threadgroup]], uint3 tg_dim [[threads_per_threadgroup]]) {
    uint row = tg_pos.x;
    uint tid = tid3.x;
    uint tg_size = tg_dim.x;
    if (row >= B) {
        return;
    }
    uint base = row * D;
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
    threadgroup float partial[256];
    threadgroup float inv_shared;
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
