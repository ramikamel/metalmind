"""MetalMind: verified Apple GPU optimization."""
from metalmind.api import analyze,optimize,OptimizationResult
def load_artifact(path, *, fallback):
    from metalmind.artifacts.loader import load_artifact as load
    return load(path, fallback=fallback)
__all__=["analyze","optimize","load_artifact","OptimizationResult"]
