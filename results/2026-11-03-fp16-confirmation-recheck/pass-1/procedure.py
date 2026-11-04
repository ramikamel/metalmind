from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time

from metalmind.agent.candidates import make_candidate
from metalmind.agent.controller import RunSummary
from metalmind.artifacts.export import finalize_run
from metalmind.artifacts.store import ArtifactStore, atomic_json
from metalmind.config import RunConfig
from metalmind.contracts import CandidateSpec, WorkloadSpec
from metalmind.runtime.environment import probe_environment
from metalmind.runtime.worker import gpu_faulted


def main():
    out = Path(__file__).resolve().parent
    pilot = Path("results/2026-11-03-assumed-fp16-pilot")
    rows = json.loads((pilot / "summary.json").read_text())["rows"]
    env = probe_environment()
    results = []
    for index, (family, case) in enumerate((("bias_gelu", "small"), ("rmsnorm", "small"), ("rmsnorm", "large"))):
        row = next(r for r in rows if (r["family"], r["case"]) == (family, case))
        assert row["outcome"] == "inconclusive"
        original = ArtifactStore(pilot / "raw" / row["run_path"])
        prior = original.read_manifest()
        assert prior["environment"] == asdict(env) and prior["phase"] == "terminal"
        workload = WorkloadSpec.model_validate(prior["workload"])
        selected = prior["archive"][0]
        candidate = CandidateSpec.model_validate(selected["candidate"])
        assert candidate.candidate_id == make_candidate(candidate.proposal, workload).candidate_id
        assert (original.path / "candidates" / f"{candidate.candidate_id}.metal").read_text() == candidate.proposal.source
        old_event, = [event for event in original.read_events()
                      if event["event"] == "evaluation" and event["candidate_id"] == candidate.candidate_id]
        assert old_event["result"]["status"] == "valid" and old_event["result"]["timing"]["domain_mode"] == "assume"
        config = RunConfig.model_validate({
            **prior["config"], "runs_dir": str(out / "jobs"), "artifacts_dir": str(out / "artifacts"),
            "execution_timeout_seconds": 300, "seed": prior["config"]["seed"] + 11003 + index,
        })
        store = ArtifactStore.create_run(config.runs_dir, config, env)
        store.write_source(candidate.candidate_id, candidate.proposal.source)
        store.write_manifest({
            **store.read_manifest(), "phase": "search_complete", "elapsed_seconds": 0,
            "wall_deadline_utc": time.time() + config.wall_seconds, "workload": prior["workload"],
            "requests": 0, "evaluated": 0, "output_tokens": 0, "archive": [selected],
            "source_run": row["run_path"],
        })
        store.append_event({**old_event, "provenance": "archived search evidence; establishes baseline requirements only"})
        print(f"Confirming {family}/{case} from the frozen pilot candidate", flush=True)
        result = finalize_run(RunSummary(
            workload=workload, config=config, run_path=store.path, outcome="revalidate",
            incumbent=candidate, incumbent_score=selected["score"],
        ))
        result_row = {"family": family, "case": case, "original_outcome": row["outcome"],
                      "original_run_path": row["run_path"], "candidate_id": candidate.candidate_id,
                      "outcome": result.outcome, "reason": result.reason, "run_path": str(result.run_path),
                      "artifact_path": str(result.artifact_path) if result.artifact_path else None, "model_requests": 0}
        results.append(result_row)
        atomic_json(out / "summary.json", {"model_requests": 0, "rows": results})
        print(json.dumps(result_row), flush=True)
        if gpu_faulted():
            subprocess.run([sys.executable, "-m", "metalmind.cli", "doctor", "--smoke", "--json"], check=True)


if __name__ == "__main__":
    main()
