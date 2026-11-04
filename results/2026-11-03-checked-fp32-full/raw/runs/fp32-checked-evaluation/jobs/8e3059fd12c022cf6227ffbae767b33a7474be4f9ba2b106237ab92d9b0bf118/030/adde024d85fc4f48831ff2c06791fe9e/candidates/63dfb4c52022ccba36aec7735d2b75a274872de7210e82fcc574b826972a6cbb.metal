#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tg_pos [[threadgroup_position_in_grid]], uint3 tp_tg [[thread_position_in_threadgroup]], uint3 t_per_tg [[threads_per_threadgroup]]) {
    (void)epsilon;
    uint row = tg_pos.x;
    uint tid = tp_tg.x;
    uint nt = t_per_tg.x;
    if (row >= B) { return; }
    threadgroup float sdata[256];
    uint base = row * D;
    float cached[8];
    uint cnt = 0;
    float local_max = -3.4028235e38f;
    for (uint col = tid; col < D; col += nt) {
        float v = x[base + col];
        if (cnt < 8u) { cached[cnt] = v; }
        cnt++;
        local_max = max(local_max, v);
    }
    bool use_cache = (cnt <= 8u);
    sdata[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = nt >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) { sdata[tid] = max(sdata[tid], sdata[tid + stride]); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = sdata[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ecached[8];
    float local_sum = 0.0f;
    if (use_cache) {
        for (uint i = 0; i < cnt; i++) {
            float e = exp(cached[i] - row_max);
            ecached[i] = e;
            local_sum += e;
        }
    } else {
        for (uint col = tid; col < D; col += nt) {
            local_sum += exp(x[base + col] - row_max);
        }
    }
    sdata[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = nt >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) { sdata[tid] += sdata[tid + stride]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / sdata[0];
    if (use_cache) {
        uint i = 0;
        for (uint col = tid; col < D; col += nt) {
            out[base + col] = ecached[i] * inv_sum;
            i++;
        }
    } else {
        for (uint col = tid; col < D; col += nt) {
            out[base + col] = exp(x[base + col] - row_max) * inv_sum;
        }
    }
}
