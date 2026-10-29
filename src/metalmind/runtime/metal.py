"""Single compatibility boundary for public PyTorch Metal shader execution.

This adapter executes trusted local source. It is not a security sandbox.
"""
from dataclasses import dataclass
from math import prod
from typing import Any
import torch


class DeviceUnavailableError(RuntimeError):
    pass


class ShaderCompilationError(RuntimeError):
    pass


@dataclass(frozen=True)
class CompiledKernel:
    function: Any
    library: Any  # Retain the library for the callable's lifetime.
    max_threads_per_threadgroup: int
    thread_execution_width: int | None = None


class MetalBackend:
    def __init__(self):
        if not torch.backends.mps.is_available():
            raise DeviceUnavailableError("A usable MPS device is required.")
        if not callable(getattr(torch.mps, "compile_shader", None)):
            raise DeviceUnavailableError("PyTorch lacks compile_shader support.")

    def compile(self, source: str, *, entry_point: str) -> CompiledKernel:
        try:
            library = torch.mps.compile_shader(source)
            function = getattr(library, entry_point)
            return CompiledKernel(function, library, int(function.max_threads_per_threadgroup))
        except (SyntaxError, RuntimeError, AttributeError) as exc:
            raise ShaderCompilationError(str(exc)) from exc

    def launch(self, kernel, inputs, output, *, threads, group_size, constants=()):
        for tensor in (*inputs, output):
            if not isinstance(tensor, torch.Tensor) or tensor.device.type != "mps":
                raise ValueError("All buffers must be MPS tensors.")
            if not tensor.is_contiguous() or tensor.storage_offset() != 0:
                raise ValueError("Buffers must be contiguous with zero storage offset.")
        for name, dims in (("threads", threads), ("threadgroup", group_size)):
            if len(dims) != 3 or any(type(v) is not int or v <= 0 for v in dims):
                raise ValueError(f"{name} dimensions must be three positive integers.")
        if prod(group_size) > kernel.max_threads_per_threadgroup:
            raise ValueError("threadgroup exceeds this pipeline's maximum.")
        kernel.function(*inputs, output, *constants, threads=threads, group_size=group_size)

    def synchronize(self):
        torch.mps.synchronize()
