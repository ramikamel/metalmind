"""Public function-level optimization entry points."""
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class OptimizationResult:
    outcome: str
    run_path: Path
    artifact_path: Path | None = None
    reason: str = ""

    def build_callable(self,*,fallback):
        from metalmind.artifacts.loader import load_artifact,OptimizedFunction
        return load_artifact(self.artifact_path,fallback=fallback) if self.artifact_path else OptimizedFunction(fallback,reason=self.reason)

def optimize(workload,*,config=None):
    from metalmind.config import RunConfig
    from metalmind.agent.controller import SearchController
    from metalmind.artifacts.export import finalize_run
    return finalize_run(SearchController().run(workload,config or RunConfig()))

def analyze(fn,example_inputs):
    from metalmind.analysis import analyze as analyze_function
    return analyze_function(fn,example_inputs)
