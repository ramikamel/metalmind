#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tgs [[threads_per_threadgroup]]) {
    if (row >= B) return;
    threadgroup float shared[256];
    device const float4* x4 = (device const float4*)x;
    device float4* out4 = (device float4*)out;
    device const float4* p4 = (device const float4*)parameter;
    uint D4 = D >> 2;
    uint base4 = row * D4;
    float sum = 0.0f;
    for (uint i = tid; i < D4; i += tgs) {
        float4 v = x4[base4 + i];
        sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    if (tid == 0) {
        uint rem = D & 3u;
        uint base = row * D + (D4 << 2);
        for (uint k = 0; k < rem; ++k) {
            float v = x[base + k];
            sum += v * v;
        }
    }
    shared[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgs >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(shared[0] / float(D) + epsilon);
    for (uint i = tid; i < D4; i += tgs) {
        float4 v = x4[base4 + i];
        float4 p = p4[i];
        out4[base4 + i] = v * inv * p;
    }
    if (tid == 0) {
        uint rem = D & 3u;
        uint base = row * D + (D4 << 2);
        uint col0 = D4 << 2;
        for (uint k = 0; k < rem; ++k) {
            out[base + k] = x[base + k] * inv * parameter[col0 + k];
        }
    }
}
