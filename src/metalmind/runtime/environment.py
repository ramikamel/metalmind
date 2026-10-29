"""Read-only capability detection; never compiles or submits work implicitly."""
import platform
from metalmind.contracts import BackendCapabilities, EnvironmentInfo


def probe_environment() -> EnvironmentInfo:
    try:
        import torch
    except ImportError:
        torch = None
    built = bool(torch is not None and torch.backends.mps.is_built())
    available = bool(torch is not None and torch.backends.mps.is_available())
    capabilities = BackendCapabilities(
        mps_built=built, mps_available=available,
        compile_shader=bool(torch is not None and callable(getattr(torch.mps, "compile_shader", None))),
        timed_events=bool(torch is not None and hasattr(torch.mps, "Event")),
    )
    reason = None
    if torch is None:
        reason = "Install MetalMind's PyTorch dependency in this Python environment."
    elif not available:
        reason = "MPS is unavailable. Use an Apple Silicon Mac with a compatible macOS/PyTorch build and GPU access."
    elif not capabilities.compile_shader:
        reason = "This PyTorch build lacks torch.mps.compile_shader; install a supported build."
    return EnvironmentInfo(1, platform.python_version(),
                           str(torch.__version__) if torch is not None else None,
                           platform.mac_ver()[0] or platform.release(),
                           platform.machine(), capabilities, reason)
