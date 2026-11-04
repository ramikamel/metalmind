#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 gid [[thread_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 tgid [[threadgroup_position_in_grid]], uint3 tgdim [[threads_per_threadgroup]]) {
    (void)gid;
    (void)epsilon;
    uint row = tgid.x;
    if (row >= B) return;
    uint lid = tid.x;
    uint nt = tgdim.x;
    device const float* rowIn = x + (uint)((ulong)row * (ulong)D);
    device float* rowOut = out + (uint)((ulong)row * (ulong)D);
    threadgroup float red[256];
    threadgroup float buf[4096];
    bool useBuf = (D <= 4096u);
    float localMax = -1.0e38f;
    for (uint i = lid; i < D; i += nt) {
        float v = rowIn[i];
        if (useBuf) buf[i] = v;
        localMax = max(localMax, v);
    }
    red[lid] = localMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = nt >> 1; stride > 0u; stride >>= 1) {
        if (lid < stride) red[lid] = max(red[lid], red[lid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rowMax = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float localSum = 0.0f;
    for (uint i = lid; i < D; i += nt) {
        float v = useBuf ? buf[i] : rowIn[i];
        float e = exp(v - rowMax);
        if (useBuf) buf[i] = e;
        localSum += e;
    }
    red[lid] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = nt >> 1; stride > 0u; stride >>= 1) {
        if (lid < stride) red[lid] += red[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float invSum = 1.0f / red[0];
    if (useBuf) {
        for (uint i = lid; i < D; i += nt) rowOut[i] = buf[i] * invSum;
    } else {
        for (uint i = lid; i < D; i += nt) rowOut[i] = exp(rowIn[i] - rowMax) * invSum;
    }
}
