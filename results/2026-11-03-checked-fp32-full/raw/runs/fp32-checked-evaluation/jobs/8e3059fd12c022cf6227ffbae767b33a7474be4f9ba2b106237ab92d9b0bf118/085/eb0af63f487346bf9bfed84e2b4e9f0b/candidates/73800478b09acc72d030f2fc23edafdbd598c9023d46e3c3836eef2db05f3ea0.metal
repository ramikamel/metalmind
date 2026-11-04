#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 gid [[thread_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]], uint3 tgsize [[threads_per_threadgroup]]) {
    uint row = tg.x;
    if (row >= B) return;
    uint lane = tid.x;
    uint T = tgsize.x;
    threadgroup float partial[256];
    ulong base = (ulong)row * (ulong)D;
    uint D4 = D >> 2;
    device const float4* x4 = (device const float4*)(x + base);
    float sum = 0.0f;
    for (uint i = lane; i < D4; i += T) {
        float4 v = x4[i];
        sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    uint tailStart = D4 << 2;
    if (lane < (D - tailStart)) {
        float v = x[base + (ulong)(tailStart + lane)];
        sum += v * v;
    }
    partial[lane] = sum;
    threadgroup_barrier(mem::mem_threadgroup);
    for (uint stride = T >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        threadgroup_barrier(mem::mem_threadgroup);
    }
    float scale = rsqrt(partial[0] / float(D) + epsilon);
    device const float4* p4 = (device const float4*)parameter;
    device float4* o4 = (device float4*)(out + base);
    for (uint i = lane; i < D4; i += T) {
        float4 v = x4[i];
        float4 p = p4[i];
        float4 r;
        r.x = v.x * scale * p.x;
        r.y = v.y * scale * p.y;
        r.z = v.z * scale * p.z;
        r.w = v.w * scale * p.w;
        o4[i] = r;
    }
    if (lane < (D - tailStart)) {
        uint idx = tailStart + lane;
        out[base + (ulong)idx] = x[base + (ulong)idx] * scale * parameter[idx];
    }
}
