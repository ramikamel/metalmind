"""Initial immutable runtime contracts. Optimization contracts follow separately."""
from dataclasses import dataclass


@dataclass(frozen=True)
class BackendCapabilities:
    mps_built: bool
    mps_available: bool
    compile_shader: bool
    timed_events: bool


@dataclass(frozen=True)
class EnvironmentInfo:
    schema_version: int
    python_version: str
    torch_version: str | None
    os_version: str
    architecture: str
    capabilities: BackendCapabilities
    unavailable_reason: str | None
