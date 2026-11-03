#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint tid [[thread_position_in_threadgroup]], uint tgid [[threadgroup_position_in_grid]], uint3 ngroups [[threadgroups_per_grid]]) {
    threadgroup float shared[256];
    uint groups = ngroups.x;
    if (groups == 0u) groups = 1u;
    for (uint row = tgid; row < B; row += groups) {
        uint base = row * D;
        float sum = 0.0f;
        if ((D & 3u) == 0u) {
            uint d4 = D >> 2;
            device const float4* x4 = (device const float4*)(x + base);
            for (uint i = tid; i < d4; i += 256u) {
                float4 v = x4[i];
                sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            }
        } else {
            for (uint i = tid; i < D; i += 256u) {
                float v = x[base + i];
                sum += v * v;
            }
        }
        shared[tid] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128u; stride > 0u; stride >>= 1u) {
            if (tid < stride) shared[tid] += shared[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float inv = rsqrt(shared[0] / float(D) + epsilon);
        if ((D & 3u) == 0u) {
            uint d4 = D >> 2;
            device const float4* x4 = (device const float4*)(x + base);
            device const float4* p4 = (device const float4*)parameter;
            device float4* o4 = (device float4*)(out + base);
            for (uint i = tid; i < d4; i += 256u) {
                float4 v = x4[i];
                float4 p = p4[i];
                o4[i] = v * inv * p;
            }
        } else {
            for (uint i = tid; i < D; i += 256u) {
                out[base + i] = x[base + i] * inv * parameter[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
