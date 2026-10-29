import pytest
import torch

from metalmind.runtime.metal import MetalBackend, ShaderCompilationError
from metalmind.runtime.health import check_health

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Real MPS device unavailable")]

SOURCE = """
#include <metal_stdlib>
using namespace metal;
kernel void transform(device const float* x [[buffer(0)]],
                      device float* y [[buffer(1)]],
                      constant uint& n [[buffer(2)]],
                      constant float& scale [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    if (i < n) y[i] = x[i] * scale + 1.0f;
}
"""


def test_transform_scalar_arguments_and_repeated_visibility():
    backend = MetalBackend()
    kernel = backend.compile(SOURCE, entry_point="transform")
    x = torch.arange(257, device="mps", dtype=torch.float32)
    out = torch.empty_like(x)
    for scale in (2.0, -3.0):
        backend.launch(kernel, (x,), out, threads=(257, 1, 1),
                       group_size=(min(64, kernel.max_threads_per_threadgroup), 1, 1),
                       constants=(257, scale))
        backend.synchronize()
        torch.testing.assert_close(out.cpu(), torch.arange(257) * scale + 1)


def test_invalid_source_is_typed_compile_failure():
    with pytest.raises(ShaderCompilationError):
        MetalBackend().compile("not Metal source", entry_point="transform")


def test_cpu_input_is_rejected_before_dispatch():
    backend = MetalBackend()
    kernel = backend.compile(SOURCE, entry_point="transform")
    with pytest.raises(ValueError, match="MPS"):
        backend.launch(kernel, (torch.ones(4),), torch.empty(4, device="mps"),
                       threads=(4, 1, 1), group_size=(4, 1, 1), constants=(4, 2.0))


def test_illegal_threadgroup_is_rejected():
    backend = MetalBackend()
    kernel = backend.compile(SOURCE, entry_point="transform")
    with pytest.raises(ValueError, match="threadgroup"):
        backend.launch(kernel, (torch.ones(4, device="mps"),), torch.empty(4, device="mps"),
                       threads=(4, 1, 1),
                       group_size=(kernel.max_threads_per_threadgroup + 1, 1, 1),
                       constants=(4, 2.0))


def test_health_probe_executes_and_validates_known_kernel():
    assert check_health()


def test_missing_entry_point_is_typed_failure():
    with pytest.raises(ShaderCompilationError):
        MetalBackend().compile(SOURCE, entry_point="does_not_exist")


def test_timed_events_bracket_completed_gpu_work():
    backend = MetalBackend()
    kernel = backend.compile(SOURCE, entry_point="transform")
    x = torch.ones(4096, device="mps")
    out = torch.empty_like(x)
    start, end = torch.mps.Event(enable_timing=True), torch.mps.Event(enable_timing=True)
    backend.synchronize()
    start.record()
    for _ in range(100):
        backend.launch(kernel, (x,), out, threads=(4096, 1, 1),
                       group_size=(64, 1, 1), constants=(4096, 2.0))
    end.record()
    end.synchronize()
    assert start.elapsed_time(end) > 0
    torch.testing.assert_close(out.cpu(), torch.full((4096,), 3.0))
