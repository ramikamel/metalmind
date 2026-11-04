#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device const half* parameter, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint tg_id [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tg_size [[threads_per_threadgroup]], uint simd_lane [[thread_index_in_simdgroup]], uint simd_id [[simdgroup_index_in_threadgroup]], uint simd_width [[threads_per_simdgroup]]) {
    if (tg_id >= B) {
        return;
    }
    uint row = tg_id;
    uint vecD = D >> 2;
    device const half4* x4 = reinterpret_cast<device const half4*>(x);
    device const half4* p4 = reinterpret_cast<device const half4*>(parameter);
    device half4* o4 = reinterpret_cast<device half4*>(out);
    uint rowVec = row * vecD;
    uint rowBase = row * D;
    float4 regs[4];
    float sum = 0.0f;
    for (uint k = 0; k < 4; ++k) {
        uint vi = tid + k * tg_size;
        if (vi < vecD) {
            float4 v = float4(x4[rowVec + vi]);
            regs[k] = v;
            sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        } else {
            regs[k] = float4(0.0f);
        }
    }
    for (uint vi = tid + 4 * tg_size; vi < vecD; vi += tg_size) {
        float4 v = float4(x4[rowVec + vi]);
        sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    for (uint col = (vecD << 2) + tid; col < D; col += tg_size) {
        float v = float(x[rowBase + col]);
        sum += v * v;
    }
    threadgroup float shared_mem[32];
    float lane_sum = simd_sum(sum);
    if (simd_lane == 0) {
        shared_mem[simd_id] = lane_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint num_simd = (tg_size + simd_width - 1) / simd_width;
    if (simd_id == 0) {
        float v = (simd_lane < num_simd) ? shared_mem[simd_lane] : 0.0f;
        float total = simd_sum(v);
        if (simd_lane == 0) {
            shared_mem[0] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = rsqrt(shared_mem[0] / float(D) + epsilon);
    for (uint k = 0; k < 4; ++k) {
        uint vi = tid + k * tg_size;
        if (vi < vecD) {
            float4 v = regs[k];
            float4 p = float4(p4[vi]);
            float4 y = v * inv * p;
            o4[rowVec + vi] = half4(half(y.x), half(y.y), half(y.z), half(y.w));
        }
    }
    for (uint vi = tid + 4 * tg_size; vi < vecD; vi += tg_size) {
        float4 v = float4(x4[rowVec + vi]);
        float4 p = float4(p4[vi]);
        float4 y = v * inv * p;
        o4[rowVec + vi] = half4(half(y.x), half(y.y), half(y.z), half(y.w));
    }
    for (uint col = (vecD << 2) + tid; col < D; col += tg_size) {
        float v = float(x[rowBase + col]);
        float p = float(parameter[col]);
        out[rowBase + col] = half(v * inv * p);
    }
}
