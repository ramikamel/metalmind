#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x [[buffer(0)]], device const float* parameter [[buffer(1)]], device float* out [[buffer(2)]], constant uint& B [[buffer(3)]], constant uint& D [[buffer(4)]], constant float& epsilon [[buffer(5)]], uint3 tg [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
    threadgroup float partial[256];
    uint row = tg.x;
    uint tid = lid.x;
    if (row >= B) {
        return;
    }
    float sum = 0.0f;
    uint base = row * D;
    for (uint col = tid; col < D; col += 256u) {
        float v = x[base + col];
        sum += v * v;
    }
    partial[tid] = sum;
    threadgroup_barrier(mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_threadgroup);
    }
    float inv = rsqrt(partial[0] / float(D) + epsilon);
    partial[0] = inv;
    threadgroup_barrier(mem_threadgroup);
    float scale = partial[0];
    for (uint col = tid; col < D; col += 256u) {
        float v = x[base + col];
        out[base + col] = float(v * scale * parameter[col]);
    }
}
