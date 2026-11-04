"""Reconfirm archived candidates after a timing failure, without model calls."""
import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("search_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    search_rows = json.loads((args.search_dir / "summary.json").read_text())["rows"]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if len(search_rows) != 18:
        parser.error("Complete the 18-case sweep before reconfirmation")
    environment = probe_environment()
    attempts = []
    for index, row in enumerate(search_rows):
        if row["outcome"] != "inconclusive" or (row["search_score"] or 0) < 1.05:
            continue
        source_id = Path(row["run_path"]).name
        original = ArtifactStore(args.search_dir / row["run_path"])
        prior = original.read_manifest()
        if prior["environment"] != asdict(environment) or prior["phase"] != "terminal":
            raise ValueError(f"Run {source_id} must be terminal and from the same environment")
        workload = WorkloadSpec.model_validate(prior["workload"])
        selected = prior["archive"][0]
        candidate = CandidateSpec.model_validate(selected["candidate"])
        source = (original.path / "candidates" / f"{candidate.candidate_id}.metal").read_text()
        identity = make_candidate(candidate.proposal, workload)
        if (candidate.candidate_id != identity.candidate_id or candidate.workload_hash != identity.workload_hash
                or source != candidate.proposal.source):
            raise ValueError(f"Run {source_id} candidate identity or source mismatch")
        event, = (event for event in original.read_events()
                  if event["event"] == "evaluation" and event["candidate_id"] == candidate.candidate_id)
        if event["result"]["status"] != "valid" or event["result"]["timing"]["domain_mode"] != "assume":
            raise ValueError(f"Run {source_id} is missing measured assumed-domain search evidence")
        for attempt in range(1):
            config = RunConfig.model_validate({
                **prior["config"],
                "runs_dir": args.output_dir / "jobs", "artifacts_dir": args.output_dir / "artifacts",
                "seed": prior["config"]["seed"] + 17301 + 4 * index + 2 * attempt,
                "warmups": 200,
            })
            store = ArtifactStore.create_run(config.runs_dir, config, environment)
            store.write_source(candidate.candidate_id, source)
            store.write_manifest({
                **store.read_manifest(), "phase": "search_complete", "elapsed_seconds": 0,
                "wall_deadline_utc": time.time() + config.wall_seconds,
                "workload": prior["workload"], "requests": 0, "evaluated": 0,
                "output_tokens": 0, "archive": [selected], "source_run": source_id,
            })
            store.append_event({**event, "provenance": "original search; establishes baseline requirements only"})
            print(f"Confirming {workload.reference_id}, {row['dtype']}/{row['case']}, attempt {attempt+1}", flush=True)
            result = finalize_run(RunSummary(
                workload=workload, config=config, run_path=store.path, outcome="revalidate",
                incumbent=candidate, incumbent_score=selected["score"],
            ))
            recorded = {
                **{key: row[key] for key in ("family", "case", "dtype")},
                "source_run": source_id, "candidate_id": candidate.candidate_id,
                "attempt": attempt + 1, "outcome": result.outcome, "reason": result.reason,
                "run_path": str(result.run_path.relative_to(args.output_dir)),
                "artifact_path": str(result.artifact_path.relative_to(args.output_dir)) if result.artifact_path else None,
                "model_requests": 0,
            }
            attempts.append(recorded)
            atomic_json(args.output_dir / "summary.json", {"model_requests": 0, "attempts": attempts})
            print(json.dumps(recorded), flush=True)
            if gpu_faulted():
                subprocess.run([sys.executable, "-m", "metalmind.cli", "doctor", "--smoke", "--json"], check=True)
            if result.outcome != "inconclusive" or result.reason != "Confirmation did not yield stable valid measurements":
                break
    atomic_json(args.output_dir / "summary.json", {"model_requests": 0, "attempts": attempts})


if __name__ == "__main__":
    main()
