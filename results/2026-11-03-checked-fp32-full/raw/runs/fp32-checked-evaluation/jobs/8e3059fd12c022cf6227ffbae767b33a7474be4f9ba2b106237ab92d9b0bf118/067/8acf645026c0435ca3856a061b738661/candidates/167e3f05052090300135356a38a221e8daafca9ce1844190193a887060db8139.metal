#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device float* out, constant uint& B, constant uint& D, constant float& epsilon, uint3 threadgroup_position [[threadgroup_position_in_grid]], uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]], uint3 threads_per_threadgroup [[threads_per_threadgroup]], uint3 threadgroups_per_grid [[threadgroups_per_grid]]) {
    (void)epsilon;
    threadgroup float cache[1024];
    threadgroup float red[256];
    const uint tid = thread_position_in_threadgroup.x;
    const uint tcount = threads_per_threadgroup.x;
    const uint ngroups = threadgroups_per_grid.x;
    for (uint row = threadgroup_position.x; row < B; row += ngroups) {
        const uint base = row * D;
        if (D == 0u) {
            continue;
        }
        if (D <= 1024u) {
            float local_max = -3.402823466e+38f;
            for (uint i = tid; i < D; i += tcount) {
                const float v = x[base + i];
                cache[i] = v;
                local_max = max(local_max, v);
            }
            red[tid] = local_max;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = tcount >> 1u; stride > 0u; stride >>= 1u) {
                if (tid < stride) {
                    red[tid] = max(red[tid], red[tid + stride]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            const float row_max = red[0];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float local_sum = 0.0f;
            for (uint i = tid; i < D; i += tcount) {
                const float e = exp(cache[i] - row_max);
                cache[i] = e;
                local_sum += e;
            }
            red[tid] = local_sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = tcount >> 1u; stride > 0u; stride >>= 1u) {
                if (tid < stride) {
                    red[tid] += red[tid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            const float inv_sum = 1.0f / red[0];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tid; i < D; i += tcount) {
                out[base + i] = cache[i] * inv_sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            float local_max = -3.402823466e+38f;
            for (uint i = tid; i < D; i += tcount) {
                local_max = max(local_max, x[base + i]);
            }
            red[tid] = local_max;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = tcount >> 1u; stride > 0u; stride >>= 1u) {
                if (tid < stride) {
                    red[tid] = max(red[tid], red[tid + stride]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            const float row_max = red[0];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float local_sum = 0.0f;
            for (uint i = tid; i < D; i += tcount) {
                local_sum += exp(x[base + i] - row_max);
            }
            red[tid] = local_sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = tcount >> 1u; stride > 0u; stride >>= 1u) {
                if (tid < stride) {
                    red[tid] += red[tid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            const float inv_sum = 1.0f / red[0];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = tid; i < D; i += tcount) {
                out[base + i] = exp(x[base + i] - row_max) * inv_sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
