#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x, device const half* parameter, device half* out, constant uint& B, constant uint& D, constant float& epsilon, uint2 gid [[thread_position_in_grid]]) {
    (void)epsilon;
    uint col = gid.x << 3u;
    uint row = gid.y << 1u;
    if (row >= B || col >= D) {
        return;
    }
    if (((D & 3u) == 0u) && (col + 8u <= D) && (row + 1u < B)) {
        float4 p0 = float4(((device const half4*)(parameter + col))[0]);
        float4 p1 = float4(((device const half4*)(parameter + col + 4u))[0]);
        for (uint r = 0u; r < 2u; ++r) {
            uint base = (row + r) * D + col;
            float4 z0 = float4(((device const half4*)(x + base))[0]) + p0;
            float4 z1 = float4(((device const half4*)(x + base + 4u))[0]) + p1;
            float4 y0 = 0.5f * z0 * (1.0f + tanh(0.79788456f * (z0 + 0.044715f * z0 * z0 * z0)));
            float4 y1 = 0.5f * z1 * (1.0f + tanh(0.79788456f * (z1 + 0.044715f * z1 * z1 * z1)));
            ((device half4*)(out + base))[0] = half4(y0);
            ((device half4*)(out + base + 4u))[0] = half4(y1);
        }
    } else {
        for (uint r = 0u; r < 2u; ++r) {
            uint rr = row + r;
            if (rr >= B) {
                break;
            }
            uint base = rr * D + col;
            for (uint i = 0u; i < 8u; ++i) {
                uint cc = col + i;
                if (cc >= D) {
                    break;
                }
                float z = float(x[base + i]) + float(parameter[cc]);
                float y = 0.5f * z * (1.0f + tanh(0.79788456f * (z + 0.044715f * z * z * z)));
                out[base + i] = half(y);
            }
        }
    }
}
