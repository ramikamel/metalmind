#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tg_pos [[threadgroup_position_in_grid]], uint3 tpig [[thread_position_in_grid]], uint3 tpitg [[thread_position_in_threadgroup]], uint3 tptg [[threads_per_threadgroup]]) {
    (void)epsilon;
    (void)tpig;
    uint row = tg_pos.x;
    if (row >= B) return;
    if (D == 0u) return;
    uint tid = tpitg.x;
    uint tg = tptg.x;
    threadgroup float sdata[256];
    device const float* row_x = x + (ulong)row * (ulong)D;
    device float* row_out = out + (ulong)row * (ulong)D;
    if ((D & 3u) == 0u) {
        uint D4 = D >> 2;
        device const float4* row_x4 = reinterpret_cast<device const float4*>(row_x);
        device float4* row_out4 = reinterpret_cast<device float4*>(row_out);
        float local_max = -3.4028234663852886e+38f;
        for (uint i = tid; i < D4; i += tg) {
            float4 v = row_x4[i];
            float m = max(max(v.x, v.y), max(v.z, v.w));
            local_max = max(local_max, m);
        }
        sdata[tid] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = tg >> 1; stride > 0u; stride >>= 1) {
            if (tid < stride) {
                float other = sdata[tid + stride];
                sdata[tid] = max(sdata[tid], other);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float row_max = sdata[0];
        float local_sum = 0.0f;
        for (uint i = tid; i < D4; i += tg) {
            float4 v = row_x4[i];
            float4 e = exp(v - row_max);
            local_sum += e.x + e.y + e.z + e.w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sdata[tid] = local_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = tg >> 1; stride > 0u; stride >>= 1) {
            if (tid < stride) sdata[tid] += sdata[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float inv_sum = 1.0f / sdata[0];
        for (uint i = tid; i < D4; i += tg) {
            float4 v = row_x4[i];
            float4 e = exp(v - row_max);
            row_out4[i] = e * inv_sum;
        }
        return;
    }
    float local_max = -3.4028234663852886e+38f;
    for (uint i = tid; i < D; i += tg) local_max = max(local_max, row_x[i]);
    sdata[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) sdata[tid] = max(sdata[tid], sdata[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = sdata[0];
    float local_sum = 0.0f;
    for (uint i = tid; i < D; i += tg) local_sum += exp(row_x[i] - row_max);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sdata[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / sdata[0];
    for (uint i = tid; i < D; i += tg) row_out[i] = exp(row_x[i] - row_max) * inv_sum;
}
