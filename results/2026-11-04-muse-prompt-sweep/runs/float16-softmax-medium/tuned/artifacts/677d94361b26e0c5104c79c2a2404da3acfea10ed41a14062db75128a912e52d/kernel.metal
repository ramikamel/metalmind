#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint gid [[thread_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint simd_id [[simdgroup_index_in_threadgroup]], uint tgid [[threadgroup_position_in_grid]], uint exec_width [[thread_execution_width]]) {
    (void)gid;
    (void)tid;
    (void)simd_id;
    (void)epsilon;
    uint row = tgid;
    if (row >= B) {
        return;
    }
    uint base = row * D;
    float4 buf[8];
    float local_max = -1.0e30f;
    for (uint i = 0u; i < 8u; ++i) {
        uint col = (i * exec_width + lane) * 4u;
        float4 v = float4(-1.0e30f);
        if (col + 3u < D) {
            v = float4(((device const half4*)(x + base + col))[0]);
        } else if (col < D) {
            if (col < D) v.x = float(x[base + col]);
            if (col + 1u < D) v.y = float(x[base + col + 1u]);
            if (col + 2u < D) v.z = float(x[base + col + 2u]);
            if (col + 3u < D) v.w = float(x[base + col + 3u]);
        }
        buf[i] = v;
        local_max = max(local_max, max(max(v.x, v.y), max(v.z, v.w)));
    }
    float row_max = simd_max(local_max);
    float local_sum = 0.0f;
    for (uint i = 0u; i < 8u; ++i) {
        uint col = (i * exec_width + lane) * 4u;
        if (col < D) {
            float4 e = exp(buf[i] - float4(row_max));
            if (col + 3u >= D) {
                if (col + 1u >= D) e.y = 0.0f;
                if (col + 2u >= D) e.z = 0.0f;
                if (col + 3u >= D) e.w = 0.0f;
            }
            buf[i] = e;
            local_sum += e.x + e.y + e.z + e.w;
        }
    }
    float inv_sum = 1.0f / simd_sum(local_sum);
    for (uint i = 0u; i < 8u; ++i) {
        uint col = (i * exec_width + lane) * 4u;
        if (col + 3u < D) {
            ((device half4*)(out + base + col))[0] = half4(buf[i] * float4(inv_sum));
        } else if (col < D) {
            float4 v = buf[i] * float4(inv_sum);
            out[base + col] = half(v.x);
            if (col + 1u < D) out[base + col + 1u] = half(v.y);
            if (col + 2u < D) out[base + col + 2u] = half(v.z);
            if (col + 3u < D) out[base + col + 3u] = half(v.w);
        }
    }
}
