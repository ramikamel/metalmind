#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tg_pos [[threadgroup_position_in_grid]], uint3 tp_tg [[thread_position_in_threadgroup]], uint3 t_per_tg [[threads_per_threadgroup]], uint simd_id [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint num_simd [[simdgroups_per_threadgroup]]) {
    (void)epsilon;
    uint row = tg_pos.x;
    if (row >= B) { return; }
    uint tid = tp_tg.x;
    uint nt = t_per_tg.x;
    uint base = row * D;
    threadgroup float red[32];
    float cached[8];
    uint cnt = 0u;
    float local_max = -3.4028235e38f;
    for (uint col = tid; col < D; col += nt) {
        float v = x[base + col];
        if (cnt < 8u) { cached[cnt] = v; }
        cnt++;
        local_max = max(local_max, v);
    }
    bool use_cache = (cnt <= 8u);
    float m = simd_max(local_max);
    if (lane == 0u) { red[simd_id] = m; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0u) {
        float v = (lane < num_simd) ? red[lane] : -3.4028235e38f;
        v = simd_max(v);
        if (lane == 0u) { red[0] = v; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ecached[8];
    float local_sum = 0.0f;
    if (use_cache) {
        for (uint i = 0u; i < cnt; i++) {
            float e = exp(cached[i] - row_max);
            ecached[i] = e;
            local_sum += e;
        }
    } else {
        for (uint col = tid; col < D; col += nt) {
            local_sum += exp(x[base + col] - row_max);
        }
    }
    float s = simd_sum(local_sum);
    if (lane == 0u) { red[simd_id] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0u) {
        float v = (lane < num_simd) ? red[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0u) { red[0] = 1.0f / v; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = red[0];
    if (use_cache) {
        uint i = 0u;
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
