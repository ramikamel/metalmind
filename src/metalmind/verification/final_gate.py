"""Single-use private verification for a frozen candidate."""
from pathlib import Path
from metalmind.contracts import EvaluationRequest
from metalmind.runtime.worker import evaluate

def run_final_gate(candidate,workload,private_seeds,*,marker_path,timeout=30):
    marker_path=Path(marker_path);marker_path.parent.mkdir(parents=True,exist_ok=True)
    with marker_path.open("x") as f: f.write(candidate.candidate_id)
    from metalmind.artifacts.store import atomic_json
    atomic_json(marker_path.with_suffix(".fixtures.json"),{"candidate_id":candidate.candidate_id,"seeds":list(private_seeds)})
    return evaluate(EvaluationRequest(candidate=candidate,workload=workload,mode="final_verify",
        private_seeds=private_seeds,compile_timeout_seconds=min(60,timeout),execution_timeout_seconds=timeout,total_timeout_seconds=timeout))
