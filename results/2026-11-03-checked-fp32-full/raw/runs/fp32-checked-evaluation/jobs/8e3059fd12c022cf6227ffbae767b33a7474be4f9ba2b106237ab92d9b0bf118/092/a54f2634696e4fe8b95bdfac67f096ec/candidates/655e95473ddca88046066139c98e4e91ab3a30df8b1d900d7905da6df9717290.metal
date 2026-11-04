#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint tid [[thread_position_in_threadgroup]], uint tg [[threadgroup_position_in_grid]], uint tpg [[threads_per_threadgroup]]) {
    if (tg >= B) {
        return;
    }
    uint row = tg;
    uint base = row * D;
    device const float* row_x = x + base;
    device float* row_o = out + base;
    uint vecD = D / 4u;
    device const float4* xv = (device const float4*)row_x;
    device const float4* pv = (device const float4*)parameter;
    device float4* ov = (device float4*)row_o;
    float sum = 0.0f;
    for (uint i = tid; i < vecD; i += tpg) {
        float4 v = xv[i];
        sum += dot(v, v);
    }
    if (tid == 0u) {
        for (uint j = vecD * 4u; j < D; ++j) {
            float v = row_x[j];
            sum += v * v;
        }
    }
    threadgroup float shared[256];
    shared[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float scale = rsqrt(shared[0] / float(D) + epsilon);
    for (uint i = tid; i < vecD; i += tpg) {
        float4 v = xv[i];
        float4 p = pv[i];
        ov[i] = v * scale * p;
    }
    if (tid == 0u) {
        for (uint j = vecD * 4u; j < D; ++j) {
            row_o[j] = row_x[j] * scale * parameter[j];
        }
    }
}
