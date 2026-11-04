#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 gid [[thread_position_in_grid]], uint3 tid3 [[thread_position_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]], uint3 tpg [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]]) {
    (void)gid;
    (void)epsilon;
    uint row = tg.x;
    if (row >= B) { return; }
    uint tid = tid3.x;
    uint nt = tpg.x;
    uint base = row * D;
    threadgroup float partial[32];
    threadgroup float shared_val;
    if (D == 1024u) {
        device const half4* x4 = (device const half4*)(x + base);
        device half4* o4 = (device half4*)(out + base);
        float4 v = float4(x4[tid]);
        float local_max = max(max(v.x, v.y), max(v.z, v.w));
        float m = simd_max(local_max);
        if (lane == 0u) { partial[sg] = m; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float pv = (lane < nsg) ? partial[lane] : -3.4028235e38f;
            float rm = simd_max(pv);
            if (lane == 0u) { shared_val = rm; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float row_max = shared_val;
        float4 e = exp(v - row_max);
        float local_sum = ((e.x + e.y) + (e.z + e.w));
        float s = simd_sum(local_sum);
        if (lane == 0u) { partial[sg] = s; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            float pv = (lane < nsg) ? partial[lane] : 0.0f;
            float rs = simd_sum(pv);
            if (lane == 0u) { shared_val = (rs > 0.0f) ? (1.0f / rs) : 0.0f; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o4[tid] = half4(e * shared_val);
        return;
    }
    float local_max = -3.4028235e38f;
    for (uint i = tid; i < D; i += nt) { local_max = max(local_max, float(x[base + i])); }
    float m = simd_max(local_max);
    if (lane == 0u) { partial[sg] = m; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float pv = (lane < nsg) ? partial[lane] : -3.4028235e38f;
        float rm = simd_max(pv);
        if (lane == 0u) { shared_val = rm; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = shared_val;
    float local_sum = 0.0f;
    for (uint i = tid; i < D; i += nt) { local_sum += exp(float(x[base + i]) - row_max); }
    float s = simd_sum(local_sum);
    if (lane == 0u) { partial[sg] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float pv = (lane < nsg) ? partial[lane] : 0.0f;
        float rs = simd_sum(pv);
        if (lane == 0u) { shared_val = (rs > 0.0f) ? (1.0f / rs) : 0.0f; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = shared_val;
    for (uint i = tid; i < D; i += nt) { out[base + i] = half(exp(float(x[base + i]) - row_max) * inv_sum); }
}
