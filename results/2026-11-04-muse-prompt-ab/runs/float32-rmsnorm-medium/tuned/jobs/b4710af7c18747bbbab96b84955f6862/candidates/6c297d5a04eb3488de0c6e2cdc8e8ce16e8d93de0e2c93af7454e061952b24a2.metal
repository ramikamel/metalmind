#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* parameter, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 tgpos [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sgroup [[simdgroup_index_in_threadgroup]], uint tptg [[threads_per_threadgroup]], uint tpsg [[threads_per_simdgroup]]) {
    uint row = tgpos.x;
    if (row >= B) { return; }
    uint col = tid * 4u;
    float4 xv = float4(0.0f);
    float4 wv = float4(0.0f);
    if (col < D) {
        device const float* xrow = x + row * D;
        if (col + 3u < D) {
            xv = ((device const float4*)(xrow + col))[0];
            wv = ((device const float4*)(parameter + col))[0];
        } else {
            for (uint i = 0u; i < 4u; ++i) {
                if (col + i < D) { xv[i] = xrow[col + i]; wv[i] = parameter[col + i]; }
            }
        }
    }
    float partial = simd_sum(dot(xv, xv));
    threadgroup float red[64];
    if (lane == 0u) { red[sgroup] = partial; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgroup == 0u) {
        uint nsimd = tpsg == 0u ? 1u : (tptg + tpsg - 1u) / tpsg;
        float v = lane < nsimd ? red[lane] : 0.0f;
        float total = simd_sum(v);
        if (lane == 0u) { red[0] = total; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = rsqrt(red[0] / float(D) + epsilon);
    float4 yv = xv * scale * wv;
    if (col < D) {
        device float* orow = out + row * D;
        if (col + 3u < D) {
            ((device float4*)(orow + col))[0] = yv;
        } else {
            for (uint i = 0u; i < 4u; ++i) {
                if (col + i < D) { orow[col + i] = yv[i]; }
            }
        }
    }
}
