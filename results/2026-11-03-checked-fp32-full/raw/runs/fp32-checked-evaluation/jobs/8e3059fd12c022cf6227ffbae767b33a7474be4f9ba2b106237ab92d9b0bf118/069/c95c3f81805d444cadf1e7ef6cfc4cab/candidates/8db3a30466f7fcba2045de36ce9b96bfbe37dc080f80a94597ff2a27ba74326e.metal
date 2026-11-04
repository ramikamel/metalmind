#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 gid [[thread_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]], uint3 tgs [[threads_per_threadgroup]]) {
    uint row = tg.x;
    uint lane = tid.x;
    uint nthreads = tgs.x;
    threadgroup float shared[256];
    float sum = 0.0f;
    uint vecN = D >> 2;
    if (row < B) {
        device const float4* x4 = reinterpret_cast<device const float4*>(x + uint(row * D));
        for (uint i = lane; i < vecN; i += nthreads) {
            float4 v = x4[i];
            sum += dot(v, v);
        }
        uint tailStart = vecN << 2;
        device const float* xr = x + uint(row * D);
        for (uint i = tailStart + lane; i < D; i += nthreads) {
            float v = xr[i];
            sum += v * v;
        }
    }
    shared[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = nthreads >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            shared[lane] += shared[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        float mean = (D > 0) ? (shared[0] / float(D)) : 0.0f;
        shared[0] = rsqrt(mean + epsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = shared[0];
    if (row < B && D > 0) {
        device const float4* x4 = reinterpret_cast<device const float4*>(x + uint(row * D));
        device const float4* p4 = reinterpret_cast<device const float4*>(parameter);
        device float4* o4 = reinterpret_cast<device float4*>(out + uint(row * D));
        for (uint i = lane; i < vecN; i += nthreads) {
            float4 v = x4[i];
            float4 p = p4[i];
            o4[i] = v * scale * p;
        }
        uint tailStart = vecN << 2;
        uint base = row * D;
        for (uint i = tailStart + lane; i < D; i += nthreads) {
            out[base + i] = x[base + i] * scale * parameter[i];
        }
    }
}
