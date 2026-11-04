#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint tg_id [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tg_dim [[threads_per_threadgroup]]) {
    (void)epsilon;
    if (tg_id >= B || D == 0u) return;
    threadgroup float shared[256];
    uint row = tg_id;
    uint base = row * D;
    float local_max = -3.402823466e+38f;
    if ((D & 3u) == 0u) {
        device const float4* x4 = reinterpret_cast<device const float4*>(x + base);
        uint d4 = D >> 2;
        for (uint i = tid; i < d4; i += tg_dim) {
            float4 v = x4[i];
            local_max = max(local_max, max(max(v.x, v.y), max(v.z, v.w)));
        }
    } else {
        for (uint i = tid; i < D; i += tg_dim) local_max = max(local_max, x[base + i]);
    }
    shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_dim >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) shared[tid] = max(shared[tid], shared[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared[0];
    float local_sum = 0.0f;
    if ((D & 3u) == 0u) {
        device const float4* x4 = reinterpret_cast<device const float4*>(x + base);
        uint d4 = D >> 2;
        for (uint i = tid; i < d4; i += tg_dim) {
            float4 e = exp(x4[i] - row_max);
            local_sum += e.x + e.y + e.z + e.w;
        }
    } else {
        for (uint i = tid; i < D; i += tg_dim) local_sum += exp(x[base + i] - row_max);
    }
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_dim >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / shared[0];
    if ((D & 3u) == 0u) {
        device const float4* x4 = reinterpret_cast<device const float4*>(x + base);
        device float4* o4 = reinterpret_cast<device float4*>(out + base);
        uint d4 = D >> 2;
        for (uint i = tid; i < d4; i += tg_dim) o4[i] = exp(x4[i] - row_max) * inv_sum;
    } else {
        for (uint i = tid; i < D; i += tg_dim) out[base + i] = exp(x[base + i] - row_max) * inv_sum;
    }
}
