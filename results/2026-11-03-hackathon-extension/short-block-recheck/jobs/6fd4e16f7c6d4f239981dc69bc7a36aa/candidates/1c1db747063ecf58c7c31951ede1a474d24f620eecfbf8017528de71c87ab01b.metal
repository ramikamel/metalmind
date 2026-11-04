#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device const half* parameter, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 gid [[thread_position_in_grid]], uint3 tg_pos [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 tg_size [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sgid [[simdgroup_index_in_threadgroup]], uint sgwidth [[threads_per_simdgroup]]) {
    threadgroup float sred[32];
    threadgroup float snorm;
    uint nsg = tg_size.x / sgwidth;
    if (D == 4096u && tg_size.x == 128u && (B & 1u) == 0u) {
        device const half4* p4 = reinterpret_cast<device const half4*>(parameter);
        uint i0 = tid.x;
        uint i1 = tid.x + 128u;
        uint i2 = tid.x + 256u;
        uint i3 = tid.x + 384u;
        uint i4 = tid.x + 512u;
        uint i5 = tid.x + 640u;
        uint i6 = tid.x + 768u;
        uint i7 = tid.x + 896u;
        half4 pv0 = p4[i0];
        half4 pv1 = p4[i1];
        half4 pv2 = p4[i2];
        half4 pv3 = p4[i3];
        half4 pv4 = p4[i4];
        half4 pv5 = p4[i5];
        half4 pv6 = p4[i6];
        half4 pv7 = p4[i7];
        for (uint rr = 0u; rr < 2u; ++rr) {
            uint row = tg_pos.x * 2u + rr;
            device const half4* x4 = reinterpret_cast<device const half4*>(x + row * 4096u);
            device half4* o4 = reinterpret_cast<device half4*>(out + row * 4096u);
            half4 xv0 = x4[i0];
            half4 xv1 = x4[i1];
            half4 xv2 = x4[i2];
            half4 xv3 = x4[i3];
            half4 xv4 = x4[i4];
            half4 xv5 = x4[i5];
            half4 xv6 = x4[i6];
            half4 xv7 = x4[i7];
            float4 f0 = float4(xv0);
            float4 f1 = float4(xv1);
            float4 f2 = float4(xv2);
            float4 f3 = float4(xv3);
            float4 f4 = float4(xv4);
            float4 f5 = float4(xv5);
            float4 f6 = float4(xv6);
            float4 f7 = float4(xv7);
            float sum = dot(f0, f0) + dot(f1, f1) + dot(f2, f2) + dot(f3, f3) + dot(f4, f4) + dot(f5, f5) + dot(f6, f6) + dot(f7, f7);
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
            o4[i0] = half4(f0 * norm * float4(pv0));
            o4[i1] = half4(f1 * norm * float4(pv1));
            o4[i2] = half4(f2 * norm * float4(pv2));
            o4[i3] = half4(f3 * norm * float4(pv3));
            o4[i4] = half4(f4 * norm * float4(pv4));
            o4[i5] = half4(f5 * norm * float4(pv5));
            o4[i6] = half4(f6 * norm * float4(pv6));
            o4[i7] = half4(f7 * norm * float4(pv7));
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        return;
    }
    for (uint rr = 0u; rr < 2u; ++rr) {
        uint row = tg_pos.x * 2u + rr;
        bool valid = row < B;
        float sum = 0.0f;
        if (valid) {
            uint base = row * D;
            uint d4 = D / 4u;
            device const half4* x4 = reinterpret_cast<device const half4*>(x + base);
            for (uint i = tid.x; i < d4; i += tg_size.x) {
                float4 v = float4(x4[i]);
                sum += dot(v, v);
            }
            for (uint i = d4 * 4u + tid.x; i < D; i += tg_size.x) {
                float v = float(x[base + i]);
                sum += v * v;
            }
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
        if (valid) {
            float norm = snorm;
            uint base = row * D;
            uint d4 = D / 4u;
            device const half4* x4 = reinterpret_cast<device const half4*>(x + base);
            device const half4* p4 = reinterpret_cast<device const half4*>(parameter);
            device half4* o4 = reinterpret_cast<device half4*>(out + base);
            for (uint i = tid.x; i < d4; i += tg_size.x) {
                o4[i] = half4(float4(x4[i]) * norm * float4(p4[i]));
            }
            for (uint i = d4 * 4u + tid.x; i < D; i += tg_size.x) {
                out[base + i] = half(float(x[base + i]) * norm * float(parameter[i]));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
