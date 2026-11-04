#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint tgpos [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sgroup [[simdgroup_index_in_threadgroup]], uint tptg [[threads_per_threadgroup]], uint tpsg [[threads_per_simdgroup]]) {
    uint row = tgpos;
    if (row >= B) return;
    uint base = row * D;
    uint idx = tid * 4u;
    float4 v = float4(0.0f);
    float4 g = float4(0.0f);
    bool valid = (idx + 3u < D) || (idx + 3u == D);
    if (idx + 3u <= D && D >= 4u) {
        v = (device const float4*)(x + base + idx)[0];
        g = (device const float4*)(parameter + idx)[0];
    }
    float partial = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    partial = simd_sum(partial);
    threadgroup float shared[32];
    if (lane == 0u) shared[sgroup] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgroup == 0u) {
        uint ngroups = tptg / tpsg;
        float p = (lane < ngroups) ? shared[lane] : 0.0f;
        p = simd_sum(p);
        if (lane == 0u) shared[0] = p;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = rsqrt(shared[0] / float(D) + epsilon);
    if (valid && idx < D) {
        float4 o = v * scale * g;
        (device float4*)(out + base + idx)[0] = o;
    }
}
