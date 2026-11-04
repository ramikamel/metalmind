#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tid3 [[thread_position_in_threadgroup]], uint3 tg3 [[threadgroup_position_in_grid]], uint3 tgs3 [[threads_per_threadgroup]]) {
    (void)epsilon;
    uint tid = tid3.x;
    uint row = tg3.x;
    uint tgsize = tgs3.x;
    if (row >= B) return;
    threadgroup float sdata[256];
    device const float* xr = x + (uint)((uint)row * D);
    device float* orow = out + (uint)((uint)row * D);
    float local_max = -INFINITY;
    if ((D & 3u) == 0u) {
        uint D4 = D >> 2;
        for (uint i = tid; i < D4; i += tgsize) {
            float4 v = ((device const float4*)xr)[i];
            local_max = max(local_max, max(max(v.x, v.y), max(v.z, v.w)));
        }
    } else {
        for (uint i = tid; i < D; i += tgsize) {
            local_max = max(local_max, xr[i]);
        }
    }
    sdata[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgsize >> 1; stride > 0u; stride >>= 1u) {
        if (tid < stride) sdata[tid] = max(sdata[tid], sdata[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = sdata[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_sum = 0.0f;
    if ((D & 3u) == 0u) {
        uint D4 = D >> 2;
        for (uint i = tid; i < D4; i += tgsize) {
            float4 v = ((device const float4*)xr)[i];
            float4 e = exp(v - row_max);
            ((device float4*)orow)[i] = e;
            local_sum += e.x + e.y + e.z + e.w;
        }
    } else {
        for (uint i = tid; i < D; i += tgsize) {
            float e = exp(xr[i] - row_max);
            orow[i] = e;
            local_sum += e;
        }
    }
    sdata[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgsize >> 1; stride > 0u; stride >>= 1u) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / sdata[0];
    if ((D & 3u) == 0u) {
        uint D4 = D >> 2;
        for (uint i = tid; i < D4; i += tgsize) {
            ((device float4*)orow)[i] *= inv_sum;
        }
    } else {
        for (uint i = tid; i < D; i += tgsize) {
            orow[i] *= inv_sum;
        }
    }
}
