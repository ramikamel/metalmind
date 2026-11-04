#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 gid [[thread_position_in_grid]], uint3 tg [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]], uint3 tgs [[threads_per_threadgroup]], uint3 grid_dim [[threads_per_grid]], uint simd_lane [[thread_index_in_simdgroup]], uint simd_id [[simdgroup_index_in_threadgroup]], uint simd_size [[threads_per_simdgroup]]) {
    threadgroup float shared[32];
    uint num_tg = grid_dim.x / max(tgs.x, 1u);
    if (num_tg == 0u) { num_tg = 1u; }
    if (D == 1024u && simd_size == 32u && tgs.x == 128u) {
        device const float4* p4 = reinterpret_cast<device const float4*>(parameter);
        float4 pv0 = p4[lid.x];
        float4 pv1 = p4[lid.x + 128u];
        for (uint row = tg.x; row < B; row += num_tg) {
            uint base = row * 1024u;
            device const float4* x4 = reinterpret_cast<device const float4*>(x + base);
            device float4* o4 = reinterpret_cast<device float4*>(out + base);
            float4 v0 = x4[lid.x];
            float4 v1 = x4[lid.x + 128u];
            float sum = dot(v0, v0) + dot(v1, v1);
            float partial = simd_sum(sum);
            if (simd_lane == 0u) { shared[simd_id] = partial; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_id == 0u) {
                float tot = (simd_lane < 4u) ? shared[simd_lane] : 0.0f;
                tot = simd_sum(tot);
                if (simd_lane == 0u) { shared[0] = rsqrt(tot * 0.0009765625f + epsilon); }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float inv = shared[0];
            o4[lid.x] = v0 * inv * pv0;
            o4[lid.x + 128u] = v1 * inv * pv1;
        }
    } else {
        uint num_simds = max(tgs.x / max(simd_size, 1u), 1u);
        for (uint row = tg.x; row < B; row += num_tg) {
            uint base = row * D;
            float sum = 0.0f;
            for (uint c = lid.x; c < D; c += tgs.x) { float v = x[base + c]; sum += v * v; }
            float partial = simd_sum(sum);
            if (simd_lane == 0u) { shared[simd_id] = partial; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lid.x == 0u) { float tot = 0.0f; for (uint i = 0u; i < num_simds; ++i) { tot += shared[i]; } shared[0] = rsqrt(tot / float(max(D, 1u)) + epsilon); }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float inv = shared[0];
            for (uint c = lid.x; c < D; c += tgs.x) { out[base + c] = x[base + c] * inv * parameter[c]; }
        }
    }
}
