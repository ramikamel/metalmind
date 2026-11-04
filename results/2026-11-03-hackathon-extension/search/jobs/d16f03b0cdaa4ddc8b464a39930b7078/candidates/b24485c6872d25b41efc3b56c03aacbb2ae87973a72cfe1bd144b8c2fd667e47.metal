#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device const half* parameter, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 gid [[thread_position_in_grid]], uint3 tg_pos [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 tg_size [[threads_per_threadgroup]], uint simd_id [[simdgroup_index_in_threadgroup]], uint simd_lane [[thread_index_in_simdgroup]], uint sg_threads [[threads_per_simdgroup]]) {
    uint lid = tid.x;
    uint nthreads = tg_size.x;
    uint row0 = tg_pos.x * 2u;
    if (row0 >= B) return;
    bool has_row1 = (row0 + 1u < B);
    threadgroup float sdata[256];
    threadgroup float partials[64];
    threadgroup float snorms[2];
    if (D == 4096u && nthreads == 256u && has_row1) {
        device const half4* x4 = reinterpret_cast<device const half4*>(x);
        device const half4* p4 = reinterpret_cast<device const half4*>(parameter);
        device half4* o4 = reinterpret_cast<device half4*>(out);
        uint base0 = row0 * 1024u;
        uint base1 = base0 + 1024u;
        uint i0 = lid;
        uint i1 = lid + 256u;
        uint i2 = lid + 512u;
        uint i3 = lid + 768u;
        half4 a0 = x4[base0 + i0];
        half4 a1 = x4[base0 + i1];
        half4 a2 = x4[base0 + i2];
        half4 a3 = x4[base0 + i3];
        half4 b0 = x4[base1 + i0];
        half4 b1 = x4[base1 + i1];
        half4 b2 = x4[base1 + i2];
        half4 b3 = x4[base1 + i3];
        float4 fa0 = float4(a0);
        float4 fa1 = float4(a1);
        float4 fa2 = float4(a2);
        float4 fa3 = float4(a3);
        float4 fb0 = float4(b0);
        float4 fb1 = float4(b1);
        float4 fb2 = float4(b2);
        float4 fb3 = float4(b3);
        float sum_a = dot(fa0, fa0) + dot(fa1, fa1) + dot(fa2, fa2) + dot(fa3, fa3);
        float sum_b = dot(fb0, fb0) + dot(fb1, fb1) + dot(fb2, fb2) + dot(fb3, fb3);
        float red_a = simd_sum(sum_a);
        float red_b = simd_sum(sum_b);
        if (simd_lane == 0u) {
            partials[simd_id * 2u] = red_a;
            partials[simd_id * 2u + 1u] = red_b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0u) {
            uint nsimd = nthreads / sg_threads;
            float total_a = 0.0f;
            float total_b = 0.0f;
            for (uint s = 0u; s < nsimd; ++s) {
                total_a += partials[s * 2u];
                total_b += partials[s * 2u + 1u];
            }
            float inv_d = 1.0f / 4096.0f;
            snorms[0] = rsqrt(total_a * inv_d + epsilon);
            snorms[1] = rsqrt(total_b * inv_d + epsilon);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float norm_a = snorms[0];
        float norm_b = snorms[1];
        half4 hp0 = p4[i0];
        half4 hp1 = p4[i1];
        half4 hp2 = p4[i2];
        half4 hp3 = p4[i3];
        float4 fp0 = float4(hp0);
        float4 fp1 = float4(hp1);
        float4 fp2 = float4(hp2);
        float4 fp3 = float4(hp3);
        o4[base0 + i0] = half4(fa0 * norm_a * fp0);
        o4[base0 + i1] = half4(fa1 * norm_a * fp1);
        o4[base0 + i2] = half4(fa2 * norm_a * fp2);
        o4[base0 + i3] = half4(fa3 * norm_a * fp3);
        o4[base1 + i0] = half4(fb0 * norm_b * fp0);
        o4[base1 + i1] = half4(fb1 * norm_b * fp1);
        o4[base1 + i2] = half4(fb2 * norm_b * fp2);
        o4[base1 + i3] = half4(fb3 * norm_b * fp3);
        return;
    }
    for (uint rr = 0u; rr < 2u; ++rr) {
        uint row = row0 + rr;
        if (row >= B) break;
        uint base = row * D;
        float sum = 0.0f;
        uint d4 = D / 4u;
        if (d4 > 0u) {
            device const half4* x4 = reinterpret_cast<device const half4*>(x + base);
            for (uint i = lid; i < d4; i += nthreads) {
                float4 v = float4(x4[i]);
                sum += dot(v, v);
            }
        }
        for (uint i = d4 * 4u + lid; i < D; i += nthreads) {
            float v = float(x[base + i]);
            sum += v * v;
        }
        if (lid < 256u) sdata[lid] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = nthreads >> 1u; stride > 0u; stride >>= 1u) {
            if (lid < stride && lid + stride < 256u) sdata[lid] += sdata[lid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lid == 0u) snorms[0] = rsqrt(sdata[0] / float(D) + epsilon);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float norm = snorms[0];
        if (d4 > 0u) {
            device const half4* x4 = reinterpret_cast<device const half4*>(x + base);
            device const half4* p4 = reinterpret_cast<device const half4*>(parameter);
            device half4* o4 = reinterpret_cast<device half4*>(out + base);
            for (uint i = lid; i < d4; i += nthreads) {
                float4 v = float4(x4[i]);
                float4 p = float4(p4[i]);
                o4[i] = half4(v * norm * p);
            }
        }
        for (uint i = d4 * 4u + lid; i < D; i += nthreads) {
            float v = float(x[base + i]);
            float p = float(parameter[i]);
            out[base + i] = half(v * norm * p);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
