#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint gid [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint simd_id [[simdgroup_index_in_threadgroup]], uint tgid [[threadgroup_position_in_grid]], uint exec_width [[thread_execution_width]]) {
    (void)gid;
    (void)epsilon;
    uint row = tgid;
    if (row >= B) {
        return;
    }
    uint base = row * D;
    float buf[8];
    float local_max = -1.0e30f;
    for (uint k = 0; k < 8; ++k) {
        uint col = tid + k * 128u;
        if (col < D) {
            float v = float(x[base + col]);
            buf[k] = v;
            local_max = max(local_max, v);
        } else {
            buf[k] = 0.0f;
        }
    }
    float simd_m = simd_max(local_max);
    threadgroup float s_max[4];
    threadgroup float s_sum[4];
    if (lane == 0u) {
        s_max[simd_id] = simd_m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0u) {
        float v = (lane < 4u) ? s_max[lane] : -1.0e30f;
        v = simd_max(v);
        if (lane == 0u) {
            s_max[0] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = s_max[0];
    float local_sum = 0.0f;
    for (uint k = 0; k < 8; ++k) {
        uint col = tid + k * 128u;
        if (col < D) {
            float e = exp(buf[k] - row_max);
            buf[k] = e;
            local_sum += e;
        }
    }
    float simd_s = simd_sum(local_sum);
    if (lane == 0u) {
        s_sum[simd_id] = simd_s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0u) {
        float v = (lane < 4u) ? s_sum[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0u) {
            s_sum[0] = 1.0f / v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = s_sum[0];
    for (uint k = 0; k < 8; ++k) {
        uint col = tid + k * 128u;
        if (col < D) {
            out[base + col] = half(buf[k] * inv_sum);
        }
    }
}
