"""Explicit opt-in live Meta generation followed by actual GPU verification."""
import json
from pathlib import Path
from metalmind.config import RunConfig
from metalmind.agent.meta import MetaProvider
from metalmind.agent.provider import ProposalContext
from metalmind.agent.candidates import make_candidate
from metalmind.workloads.registry import workload_for_shape
from metalmind.runtime.worker import evaluate
from metalmind.contracts import EvaluationRequest


def main():
    workload=workload_for_shape("bias_gelu",(2,3))
    response=MetaProvider(RunConfig()).propose(ProposalContext(workload=workload,remaining_requests=1))
    if response.proposal is None: raise SystemExit(response.error)
    candidate=make_candidate(response.proposal,workload,model=response.model)
    result=evaluate(EvaluationRequest(candidate=candidate,workload=workload))
    folder=Path("runs/first-generation");folder.mkdir(parents=True,exist_ok=True)
    (folder/"candidate.json").write_text(candidate.model_dump_json(indent=2))
    (folder/"result.json").write_text(result.model_dump_json(indent=2))
    print(json.dumps({"model":response.model,"status":result.status,
        "verification":result.verification.model_dump() if result.verification else None,
        "reason":result.reason,"usage":response.usage.model_dump()},indent=2))
    return 0 if result.status=="valid" else 1

if __name__=="__main__": raise SystemExit(main())
