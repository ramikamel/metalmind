#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 gid [[thread_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]], uint3 tgid [[threadgroup_position_in_grid]], uint3 tgdim [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sgid [[simdgroup_index_in_threadgroup]], uint simdWidth [[thread_execution_width]]) {
    (void)gid;
    (void)epsilon;
    uint row = tgid.x;
    if (row >= B) return;
    uint lid = tid.x;
    uint nt = tgdim.x;
    uint numSg = nt / simdWidth;
    device const float* rowIn = x + (ulong)row * (ulong)D;
    device float* rowOut = out + (ulong)row * (ulong)D;
    threadgroup float red[64];
    threadgroup float bcast;
    uint numVec = D >> 2;
    uint tail = D & 3u;
    float localMax = -1.0e38f;
    for (uint i = lid; i < numVec; i += nt) {
        float4 v = ((device const float4*)rowIn)[i];
        localMax = max(localMax, max(max(v.x, v.y), max(v.z, v.w)));
    }
    if (tail > 0u && lid < tail) {
        localMax = max(localMax, rowIn[numVec * 4u + lid]);
    }
    float sgMax = simd_max(localMax);
    if (lane == 0u) red[sgid] = sgMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        float v = (lane < numSg) ? red[lane] : -1.0e38f;
        v = simd_max(v);
        if (lane == 0u) bcast = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rowMax = bcast;
    float localSum = 0.0f;
    for (uint i = lid; i < numVec; i += nt) {
        float4 v = ((device const float4*)rowIn)[i];
        float4 e = exp(v - rowMax);
        ((device float4*)rowOut)[i] = e;
        localSum += e.x + e.y + e.z + e.w;
    }
    if (tail > 0u && lid < tail) {
        uint idx = numVec * 4u + lid;
        float e = exp(rowIn[idx] - rowMax);
        rowOut[idx] = e;
        localSum += e;
    }
    float sgSum = simd_sum(localSum);
    if (lane == 0u) red[sgid] = sgSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        float v = (lane < numSg) ? red[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0u) bcast = 1.0f / v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float invSum = bcast;
    for (uint i = lid; i < numVec; i += nt) {
        float4 e = ((device float4*)rowOut)[i];
        ((device float4*)rowOut)[i] = e * invSum;
    }
    if (tail > 0u && lid < tail) {
        uint idx = numVec * 4u + lid;
        rowOut[idx] = rowOut[idx] * invSum;
    }
}
