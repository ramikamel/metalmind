#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device const half* parameter, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 gid [[thread_position_in_grid]], uint3 tg_pos [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 tg_size [[threads_per_threadgroup]], uint3 grid_size [[threads_per_grid]], uint lane [[thread_index_in_simdgroup]], uint sgid [[simdgroup_index_in_threadgroup]], uint sgwidth [[threads_per_simdgroup]]) {
    threadgroup float sred[32];
    threadgroup float snorm;
    uint nsg = tg_size.x / sgwidth;
    uint ntg = grid_size.x / tg_size.x;
    if (D == 4096u && tg_size.x == 256u && (B & 3u) == 0u) {
        uint i0 = tid.x;
        uint i1 = tid.x + 256u;
        device const uint4* pU = reinterpret_cast<device const uint4*>(parameter);
        uint4 pr0 = pU[i0];
        uint4 pr1 = pU[i1];
        float4 pf0 = float4(as_type<half4>(pr0.xy));
        float4 pf1 = float4(as_type<half4>(pr0.zw));
        float4 pf2 = float4(as_type<half4>(pr1.xy));
        float4 pf3 = float4(as_type<half4>(pr1.zw));
        for (uint chunk = tg_pos.x * 4u; chunk < B; chunk += ntg * 4u) {
            for (uint rr = 0u; rr < 4u; ++rr) {
                uint row = chunk + rr;
                device const uint4* xU = reinterpret_cast<device const uint4*>(x + row * 4096u);
                device uint4* oU = reinterpret_cast<device uint4*>(out + row * 4096u);
                uint4 xr0 = xU[i0];
                uint4 xr1 = xU[i1];
                float4 f0 = float4(as_type<half4>(xr0.xy));
                float4 f1 = float4(as_type<half4>(xr0.zw));
                float4 f2 = float4(as_type<half4>(xr1.xy));
                float4 f3 = float4(as_type<half4>(xr1.zw));
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
                uint4 ov0;
                uint4 ov1;
                ov0.xy = as_type<uint2>(half4(f0 * norm * pf0));
                ov0.zw = as_type<uint2>(half4(f1 * norm * pf1));
                ov1.xy = as_type<uint2>(half4(f2 * norm * pf2));
                ov1.zw = as_type<uint2>(half4(f3 * norm * pf3));
                oU[i0] = ov0;
                oU[i1] = ov1;
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        return;
    }
    for (uint row = tg_pos.x; row < B; row += ntg) {
        uint base = row * D;
        float sum = 0.0f;
        for (uint i = tid.x; i < D; i += tg_size.x) {
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
        for (uint i = tid.x; i < D; i += tg_size.x) {
            out[base + i] = half(float(x[base + i]) * norm * float(parameter[i]));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
