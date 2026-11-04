#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device const half* parameter, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 gid [[thread_position_in_grid]], uint3 tg_pos [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 tg_size [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sgid [[simdgroup_index_in_threadgroup]], uint sgwidth [[threads_per_simdgroup]]) {
    threadgroup float sred[32];
    threadgroup float snorm;
    uint nsg = tg_size.x / sgwidth;
    if (D == 4096u && tg_size.x == 256u && (B & 3u) == 0u) {
        device const half4* p4 = reinterpret_cast<device const half4*>(parameter);
        uint i0 = tid.x;
        uint i1 = tid.x + 256u;
        uint i2 = tid.x + 512u;
        uint i3 = tid.x + 768u;
        half4 pv0 = p4[i0];
        half4 pv1 = p4[i1];
        half4 pv2 = p4[i2];
        half4 pv3 = p4[i3];
        float4 pw0 = float4(pv0);
        float4 pw1 = float4(pv1);
        float4 pw2 = float4(pv2);
        float4 pw3 = float4(pv3);
        for (uint rr = 0u; rr < 4u; ++rr) {
            uint row = tg_pos.x * 4u + rr;
            device const half4* x4 = reinterpret_cast<device const half4*>(x + row * 4096u);
            device half4* o4 = reinterpret_cast<device half4*>(out + row * 4096u);
            float4 f0 = float4(x4[i0]);
            float4 f1 = float4(x4[i1]);
            float4 f2 = float4(x4[i2]);
            float4 f3 = float4(x4[i3]);
            float sum = dot(f0, f0) + dot(f1, f1) + dot(f2, f2) + dot(f3, f3);
            float ssum = simd_sum(sum);
            if (lane == 0u) sred[sgid] = ssum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid.x == 0u) {
                float tot = 0.0f;
                for (uint s = 0u; s < nsg; ++s) tot += sred[s];
                snorm = rsqrt(tot * (1.0f / 4096.0f) + epsilon);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float norm = snorm;
            o4[i0] = half4(f0 * norm * pw0);
            o4[i1] = half4(f1 * norm * pw1);
            o4[i2] = half4(f2 * norm * pw2);
            o4[i3] = half4(f3 * norm * pw3);
        }
        return;
    }
    uint ngroups = gid.x;
    ngroups = (tg_pos.x > 0u || tg_size.x > 0u) ? (256u) : 256u;
    for (uint row = tg_pos.x; row < B; row += ngroups) {
        uint base = row * D;
        uint d4 = D / 4u;
        device const half4* x4 = reinterpret_cast<device const half4*>(x + base);
        float sum = 0.0f;
        for (uint i = tid.x; i < d4; i += tg_size.x) {
            float4 v = float4(x4[i]);
            sum += dot(v, v);
        }
        for (uint i = d4 * 4u + tid.x; i < D; i += tg_size.x) {
            float v = float(x[base + i]);
            sum += v * v;
        }
        float ssum = simd_sum(sum);
        if (lane == 0u) sred[sgid] = ssum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid.x == 0u) {
            float tot = 0.0f;
            for (uint s = 0u; s < nsg; ++s) tot += sred[s];
            snorm = rsqrt(tot / float(D) + epsilon);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float norm = snorm;
        device const half4* p4 = reinterpret_cast<device const half4*>(parameter);
        device half4* o4 = reinterpret_cast<device half4*>(out + base);
        for (uint i = tid.x; i < d4; i += tg_size.x) {
            o4[i] = half4(float4(x4[i]) * norm * float4(p4[i]));
        }
        for (uint i = d4 * 4u + tid.x; i < D; i += tg_size.x) {
            out[base + i] = half(float(x[base + i]) * norm * float(parameter[i]));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
