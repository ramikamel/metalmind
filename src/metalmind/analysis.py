"""Explain supported selected functions and distinguish estimates from measurements."""
from metalmind.contracts import Contract,WorkloadSpec
from metalmind.workloads.capture import capture_function,UnsupportedReason

class AnalysisResult(Contract):
    supported: bool
    reason: str | None = None
    workload: WorkloadSpec | None = None
    estimated_input_bytes: int = 0
    hypothesis: str | None = None

def analyze(fn,example_inputs):
    captured=capture_function(fn,example_inputs)
    if isinstance(captured,UnsupportedReason): return AnalysisResult(supported=False,reason=captured.reason)
    traffic=sum(t.numel()*t.element_size() for t in example_inputs)
    return AnalysisResult(supported=True,workload=captured,estimated_input_bytes=traffic,
        hypothesis="Fusion may remove intermediate tensor traffic and dispatch overhead; benchmark evidence is required.")
