#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 gid [[thread_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]], uint3 tgs [[threads_per_threadgroup]], uint simd_lane [[thread_index_in_simdgroup]], uint simd_id [[simdgroup_index_in_threadgroup]], uint num_simd [[simdgroups_per_threadgroup]]) {
    uint row = tg.x;
    uint lane = tid.x;
    uint nthreads = tgs.x;
    threadgroup float shared[64];
    float4 xv[4];
    float4 pv[4];
    float sum = 0.0f;
    uint vecN = D >> 2;
    if (row < B) {
        device const float4* x4 = reinterpret_cast<device const float4*>(x + uint(row * D));
        device const float4* p4 = reinterpret_cast<device const float4*>(parameter);
        for (uint k = 0; k < 4; ++k) {
            uint i = lane + k * nthreads;
            if (i < vecN) {
                float4 v = x4[i];
                xv[k] = v;
                pv[k] = p4[i];
                sum += dot(v, v);
            } else {
                xv[k] = float4(0.0f);
                pv[k] = float4(0.0f);
            }
        }
        uint tailStart = vecN << 2;
        if (tailStart < D) {
            device const float* xr = x + uint(row * D);
            for (uint i = tailStart + lane; i < D; i += nthreads) {
                float v = xr[i];
                sum += v * v;
            }
        }
    } else {
        for (uint k = 0; k < 4; ++k) { xv[k] = float4(0.0f); pv[k] = float4(0.0f); }
    }
    float simd_total = simd_sum(sum);
    if (simd_lane == 0) {
        shared[simd_id] = simd_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        float total = 0.0f;
        for (uint i = 0; i < num_simd; ++i) {
            total += shared[i];
        }
        float mean = (D > 0) ? (total / float(D)) : 0.0f;
        shared[0] = rsqrt(mean + epsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = shared[0];
    if (row < B && D > 0) {
        device float4* o4 = reinterpret_cast<device float4*>(out + uint(row * D));
        float s = scale;
        for (uint k = 0; k < 4; ++k) {
            uint i = lane + k * nthreads;
            if (i < vecN) {
                o4[i] = xv[k] * s * pv[k];
            }
        }
        uint tailStart = vecN << 2;
        if (tailStart < D) {
            uint base = row * D;
            for (uint i = tailStart + lane; i < D; i += nthreads) {
                out[base + i] = x[base + i] * s * parameter[i];
            }
        }
    }
}
