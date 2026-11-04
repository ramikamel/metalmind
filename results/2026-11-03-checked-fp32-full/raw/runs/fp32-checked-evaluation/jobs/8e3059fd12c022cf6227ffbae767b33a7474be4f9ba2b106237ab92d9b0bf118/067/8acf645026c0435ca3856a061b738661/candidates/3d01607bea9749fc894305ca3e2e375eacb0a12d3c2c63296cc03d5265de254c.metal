#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tg [[threads_per_threadgroup]]) {
    (void)epsilon;
    if (row >= B || D == 0) return;
    device const float* xr = x + (uint)row * D;
    device float* orow = out + (uint)row * D;
    threadgroup float sh[256];
    float lmax = -INFINITY;
    for (uint i = tid; i < D; i += tg) { lmax = max(lmax, xr[i]); }
    sh[tid] = lmax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg >> 1; s > 0; s >>= 1) { if (tid < s) sh[tid] = max(sh[tid], sh[tid + s]); threadgroup_barrier(mem_flags::mem_threadgroup); }
    float rmax = sh[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float lsum = 0.0f;
    for (uint i = tid; i < D; i += tg) { lsum += exp(xr[i] - rmax); }
    sh[tid] = lsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg >> 1; s > 0; s >>= 1) { if (tid < s) sh[tid] += sh[tid + s]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float inv = 1.0f / sh[0];
    for (uint i = tid; i < D; i += tg) { orow[i] = (float)(exp(xr[i] - rmax) * inv); }
}