#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint thread_position_in_grid [[thread_position_in_grid]], uint threadgroup_position_in_grid [[threadgroup_position_in_grid]], uint thread_position_in_threadgroup [[thread_position_in_threadgroup]], uint threads_per_threadgroup [[threads_per_threadgroup]]) {
    (void)thread_position_in_grid;
    (void)epsilon;
    uint row = threadgroup_position_in_grid;
    uint tid = thread_position_in_threadgroup;
    uint tcount = threads_per_threadgroup;
    if (row >= B) { return; }
    uint base = row * D;
    threadgroup float sdata[256];
    float v[16];
    float local_max = -3.4028234663852886e+38f;
    for (uint k = 0; k < 16; ++k) {
        uint idx = k * tcount + tid;
        float val = -3.4028234663852886e+38f;
        if (idx < D) { val = x[base + idx]; }
        v[k] = val;
        local_max = max(local_max, val);
    }
    if (tid < 256) { sdata[tid] = local_max; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) { sdata[tid] = max(sdata[tid], sdata[tid + s]); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = sdata[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_sum = 0.0f;
    for (uint k = 0; k < 16; ++k) {
        uint idx = k * tcount + tid;
        float e = 0.0f;
        if (idx < D) { e = exp(v[k] - row_max); }
        v[k] = e;
        local_sum += e;
    }
    if (tid < 256) { sdata[tid] = local_sum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) { sdata[tid] += sdata[tid + s]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / sdata[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < 16; ++k) {
        uint idx = k * tcount + tid;
        if (idx < D) { out[base + idx] = v[k] * inv_sum; }
    }
}
