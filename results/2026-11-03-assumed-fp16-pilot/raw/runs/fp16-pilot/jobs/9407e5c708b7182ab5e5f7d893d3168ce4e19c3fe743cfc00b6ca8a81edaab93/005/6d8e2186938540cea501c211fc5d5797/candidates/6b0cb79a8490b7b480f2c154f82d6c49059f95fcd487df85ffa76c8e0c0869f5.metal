#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tptg [[threads_per_threadgroup]], uint sg_id [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg_size [[threads_per_simdgroup]]) {
    (void)epsilon;
    if (row >= B) return;
    device const half* in_row = x + (uint)row * D;
    device half* out_row = out + (uint)row * D;
    threadgroup float sbuf[32];
    uint num_sg = tptg / sg_size;
    float vals[32];
    uint cnt = 0;
    float local_max = -1.0e30f;
    for (uint i = tid; i < D; i += tptg) {
        float v = float(in_row[i]);
        if (cnt < 32) vals[cnt] = v;
        cnt++;
        local_max = fmax(local_max, v);
    }
    float sg_max = simd_max(local_max);
    if (lane == 0) sbuf[sg_id] = sg_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id == 0) {
        float v = (lane < num_sg) ? sbuf[lane] : -1.0e30f;
        v = simd_max(v);
        if (lane == 0) sbuf[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = sbuf[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool cached = (cnt <= 32);
    float local_sum = 0.0f;
    if (cached) {
        for (uint k = 0; k < cnt; k++) {
            float e = exp2((vals[k] - row_max) * 1.4426950408889634f);
            vals[k] = e;
            local_sum += e;
        }
    } else {
        for (uint i = tid; i < D; i += tptg) local_sum += exp2((float(in_row[i]) - row_max) * 1.4426950408889634f);
    }
    float sg_sum = simd_sum(local_sum);
    if (lane == 0) sbuf[sg_id] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id == 0) {
        float v = (lane < num_sg) ? sbuf[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) sbuf[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = 1.0f / sbuf[0];
    if (cached) {
        uint k = 0;
        for (uint i = tid; i < D; i += tptg) {
            out_row[i] = half(vals[k] * inv_sum);
            k++;
        }
    } else {
        for (uint i = tid; i < D; i += tptg) out_row[i] = half(exp2((float(in_row[i]) - row_max) * 1.4426950408889634f) * inv_sum);
    }
}
