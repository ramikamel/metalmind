"""Known-good compilation, dispatch, synchronization, and output probe."""
import torch
from metalmind.runtime.metal import MetalBackend, DeviceUnavailableError, ShaderCompilationError

_SOURCE = """
#include <metal_stdlib>
using namespace metal;
kernel void health(device float* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    out[i] = float(i) + 1.0f;
}
"""


def check_health() -> bool:
    try:
        backend = MetalBackend()
        kernel = backend.compile(_SOURCE, entry_point="health")
        out = torch.empty(16, device="mps", dtype=torch.float32)
        backend.launch(kernel, (), out, threads=(16, 1, 1), group_size=(16, 1, 1))
        backend.synchronize()
        return bool(torch.equal(out.cpu(), torch.arange(1, 17, dtype=torch.float32)))
    except (DeviceUnavailableError, ShaderCompilationError, RuntimeError):
        return False
