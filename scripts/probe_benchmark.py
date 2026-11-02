"""Benchmark the existing verified candidate; makes no model API request."""
import json
from pathlib import Path
from metalmind.contracts import CandidateSpec,EvaluationRequest
from metalmind.runtime.worker import evaluate
from metalmind.workloads.registry import workload_for_shape

if __name__=='__main__':
    candidate=CandidateSpec.model_validate_json(Path('runs/first-generation/candidate.json').read_text())
    workload=workload_for_shape('bias_gelu',(2,3))
    result=evaluate(EvaluationRequest(candidate=candidate,workload=workload,measure=True,execution_timeout_seconds=90))
    Path('runs/first-generation/benchmark.json').write_text(result.model_dump_json(indent=2))
    print(result.model_dump_json(indent=2))
