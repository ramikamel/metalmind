"""Run the separately scoped higher-budget model search on the local GPU."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

from metalmind.agent.controller import SearchController
from metalmind.agent.meta import MetaProvider
from metalmind.artifacts.export import finalize_run
from metalmind.artifacts.store import atomic_json
from metalmind.config import RunConfig
from metalmind.runtime.worker import gpu_faulted
from metalmind.workloads.registry import get_workload


TARGETS = (("rmsnorm", "large", "float16"), ("softmax", "medium", "float32"),
           ("rmsnorm", "medium", "float32"), ("softmax", "large", "float32"))


class PriorEvidenceProvider:
    def __init__(self, provider, prior_source):
        self.provider = provider
        self.prior_source = prior_source

    def propose(self, context):
        history = (
            "Prior archived FP16 Metal kernel passed correctness. It ran about 17–18x faster than eager but tied "
            "the verified Inductor baseline at roughly 0.05 ms per call on this 14-core M3 Pro. It was not promoted. "
            "Seek a materially different schedule that beats Inductor while retaining FP32 intermediate semantics. "
            "The prior source is a starting point. Any new incumbent and feedback take priority."
        )
        return self.provider.propose(context.model_copy(update={
            "feedback": (history, *context.feedback),
            "incumbent_source": context.incumbent_source or self.prior_source,
        }))


def main():
    out = Path(__file__).resolve().parent
    evidence = Path("results/2026-11-03-fp16-large-rmsnorm-resolution/session-1-attempt-1-request.json")
    prior = json.loads(evidence.read_text())["candidate"]["proposal"]["source"]
    atomic_json(out/"protocol.json", {
        "targets": TARGETS, "domain_mode": "assume", "model": "muse-wonder-1.0",
        "reasoning_effort": "medium", "max_requests_per_target": 6, "max_candidates_per_target": 6,
        "max_output_tokens_per_request": 8192, "max_total_requests": 24,
        "execution_timeout_seconds": 420, "prior_fp16_evidence": str(evidence),
    })
    rows=[]
    for index,(family,case,dtype) in enumerate(TARGETS):
        name=f"{dtype}-{family}-{case}"
        workload=get_workload(family,case,dtype)
        config=RunConfig(domain_mode="assume", reasoning_effort="medium",
            max_requests=6,max_candidates=6,max_output_tokens=8192,total_output_tokens=49152,
            wall_seconds=3600,finalization_reserve_seconds=1100,execution_timeout_seconds=420,
            request_timeout_seconds=180,compile_timeout_seconds=90,seed=202611030+index,
            runs_dir=out/"jobs",artifacts_dir=out/"artifacts")
        provider=MetaProvider(config)
        if dtype=="float16":
            provider=PriorEvidenceProvider(provider,prior)
        print(f"Starting {name}",flush=True)
        summary=SearchController(provider=provider).run(workload,config)
        result=finalize_run(summary)
        row={**asdict(result),"family":family,"case":case,"dtype":dtype,
             "requests":summary.requests,"evaluated":summary.evaluated,"search_score":summary.incumbent_score}
        row={k:str(v) if isinstance(v,Path) else v for k,v in row.items()}
        rows.append(row)
        atomic_json(out/"summary.json",{"rows":rows,"requests":sum(r["requests"] for r in rows)})
        print(json.dumps(row),flush=True)
        if gpu_faulted():
            subprocess.run([sys.executable,"-m","metalmind.cli","doctor","--smoke","--json"],check=True)


if __name__=="__main__":
    main()
