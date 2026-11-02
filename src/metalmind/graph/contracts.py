from metalmind.contracts import Contract, TensorSpec, WorkloadSpec

class LatencySamples(Contract):
    median_ms: float
    samples_ms: tuple[float, ...]
    repetitions: int
    boundary: str = "standalone synchronized public callable; not additive attribution"

class RegionInfo(Contract):
    region_id: str
    family: str
    output_name: str
    node_names: tuple[str, ...]
    input_names: tuple[str, ...]
    inputs: tuple[TensorSpec, ...]
    workload: WorkloadSpec | None = None
    estimated_input_bytes: int
    reason: str | None = None
    latency: LatencySamples | None = None

class ModuleAnalysis(Contract):
    supported: bool
    reason: str | None = None
    graph_hash: str | None = None
    inputs: tuple[TensorSpec, ...] = ()
    regions: tuple[RegionInfo, ...] = ()
    exclusions: tuple[str, ...] = ()
    whole_latency: LatencySamples | None = None
    ranking_basis: str = "estimated input bytes; no timing attribution"
    constraints: tuple[str, ...] = ("flat pure eval-mode FP32 modules", "standalone times must not be summed",)
