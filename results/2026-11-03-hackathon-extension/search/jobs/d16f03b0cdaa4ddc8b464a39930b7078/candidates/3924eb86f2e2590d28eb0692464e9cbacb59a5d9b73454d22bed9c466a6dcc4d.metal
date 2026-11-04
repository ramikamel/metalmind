#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device const half* parameter, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 gid [[thread_position_in_grid]], uint3 tg_pos [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 tg_size [[threads_per_threadgroup]], uint3 num_tg [[threadgroups_per_grid]], uint lane [[thread_index_in_simdgroup]], uint sgid [[simdgroup_index_in_threadgroup]], uint simd_w [[threads_per_simdgroup]]) {
    uint lid = tid.x;
    uint nthreads = tg_size.x;
    if (D == 4096u && nthreads == 256u) {
        threadgroup half4 sp[1024];
        threadgroup float sred[32];
        threadgroup float snorm[4];
        device const half4* p4g = reinterpret_cast<device const half4*>(parameter);
        for (uint i = lid; i < 1024u; i += 256u) sp[i] = p4g[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint sub = lid >> 6u;
        uint col = lid & 63u;
        uint perRow = (simd_w > 0u) ? (64u / simd_w) : 1u;
        for (uint batch = tg_pos.x * 4u; batch < B; batch += num_tg.x * 4u) {
            uint row = batch + sub;
            bool valid = row < B;
            uint base4 = row * 1024u;
            half4 v[16];
            float sum = 0.0f;
            for (uint k = 0u; k < 16u; ++k) {
                uint c = col + k * 64u;
                half4 hv = valid ? reinterpret_cast<device const half4*>(x)[base4 + c] : half4(0.0h);
                v[k] = hv;
                float4 fv = float4(hv);
                sum += dot(fv, fv);
            }
            sum = simd_sum(sum);
            if (lane == 0u) sred[sgid] = sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lid < 4u) {
                float s = 0.0f;
                for (uint j = 0u; j < perRow; ++j) s += sred[lid * perRow + j];
                snorm[lid] = rsqrt(s * (1.0f / 4096.0f) + epsilon);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (valid) {
                float norm = snorm[sub];
                device half4* o4 = reinterpret_cast<device half4*>(out);
                for (uint k = 0u; k < 16u; ++k) {
                    uint c = col + k * 64u;
                    o4[base4 + c] = half4(float4(v[k]) * norm * float4(sp[c]));
                }
            }
        }
        return;
    }
    threadgroup float sdata[256];
    threadgroup float snorm1;
    for (uint row = tg_pos.x; row < B; row += num_tg.x) {
        uint base = row * D;
        float sum = 0.0f;
        uint D4 = D / 4u;
        if (D4 > 0u) {
            device const half4* x4 = reinterpret_cast<device const half4*>(x + base);
            for (uint i = lid; i < D4; i += nthreads) { float4 fv = float4(x4[i]); sum += dot(fv, fv); }
        }
        for (uint i = D4 * 4u + lid; i < D; i += nthreads) { float fv = float(x[base + i]); sum += fv * fv; }
        if (lid < 256u) sdata[lid] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = nthreads >> 1u; stride > 0u; stride >>= 1u) {
            if (lid < stride) sdata[lid] += sdata[lid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lid == 0u) snorm1 = rsqrt(sdata[0] / float(D) + epsilon);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float norm = snorm1;
        if (D4 > 0u) {
            device const half4* x4 = reinterpret_cast<device const half4*>(x + base);
            device const half4* p4 = reinterpret_cast<device const half4*>(parameter);
            device half4* o4 = reinterpret_cast<device half4*>(out + base);
            for (uint i = lid; i < D4; i += nthreads) o4[i] = half4(float4(x4[i]) * norm * float4(p4[i]));
        }
        for (uint i = D4 * 4u + lid; i < D; i += nthreads) out[base + i] = half(float(x[base + i]) * norm * float(parameter[i]));
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}