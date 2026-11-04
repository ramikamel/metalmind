"""Diagnose the softmax race and test a targeted Muse prompt on that case."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import runpy
from unittest.mock import patch

from metalmind.agent import meta
from metalmind.agent.candidates import make_candidate
from metalmind.agent.controller import SearchController
from metalmind.artifacts.export import finalize_run, verify_artifact_evidence
from metalmind.artifacts.store import atomic_json
from metalmind.config import RunConfig
from metalmind.contracts import CandidateSpec, EvaluationRequest, WorkloadSpec
from metalmind.runtime.worker import evaluate, gpu_faulted

ADVICE = (
    "The supplied incumbent sometimes changes its output on identical inputs. "
    "Inspect shared scratch reuse across SIMD groups. In particular, a threadgroup barrier "
    "before reading a shared maximum does not prevent a fast SIMD group overwriting that "
    "address with a sum while another SIMD group is still reading the maximum. "
    "Use distinct shared buffers for those phases or add a uniform barrier after all readers "
    "consume the shared value and before any writer overwrites it. Keep the workload, FP32 "
    "intermediates and vectorized output. Explain the synchronization change in hypothesis."
)

def read(path):
    return json.loads(path.read_text())

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    args.experiment = args.experiment.resolve()
    common = runpy.run_path(str(args.experiment / "procedure.py"))
    rows = read(args.experiment / "summary.json")["rows"]
    assert len(rows) == 8, "Complete the A/B test before running the diagnostic"
    target = next(row for row in rows if row["arm"]=="tuned" and row["family"]=="softmax")
    job = args.experiment / target["run_path"]
    assert read(job / "private-verification.json")["verification"]["reason"]=="nondeterministic output"
    manifest = read(job / "manifest.json")
    original = CandidateSpec.model_validate(manifest["archive"][0]["candidate"])
    workload = WorkloadSpec.model_validate(manifest["workload"])
    seeds = tuple(read(job / "private-gate-consumed.fixtures.json")["seeds"])
    pattern = "float rmax = sbuf[0];"
    assert original.proposal.source.count(pattern)==2
    patched_source = re.sub(
        r"^( +)float rmax = sbuf\[0\];$",
        r"\g<0>\n\g<1>threadgroup_barrier(mem_flags::mem_threadgroup);",
        original.proposal.source, flags=re.MULTILINE,
    )
    assert patched_source.count("threadgroup_barrier(") == original.proposal.source.count("threadgroup_barrier(") + 2
    patched = make_candidate(original.proposal.model_copy(update={
        "source": patched_source, "hypothesis": "Wait for shared maximum readers before reusing scratch."
    }), workload, parent_id=original.candidate_id, model="manual-race-control")
    root.mkdir(parents=True, exist_ok=False)
    (root / "manual-barrier-control.metal").write_text(patched_source)
    diagnostic = {}
    for label,candidate in (("original",original),("barrier_control",patched)):
        outcomes = []
        for _ in range(3):
            result = evaluate(EvaluationRequest(candidate=candidate, workload=workload,
                mode="final_verify", private_seeds=seeds, compile_timeout_seconds=60,
                execution_timeout_seconds=60, total_timeout_seconds=90))
            outcomes.append(result.model_dump(mode="json"))
            if gpu_faulted():
                raise RuntimeError("GPU fault during race diagnosis")
        diagnostic[label] = outcomes
        print(label, [(row["status"], (row["verification"] or {}).get("reason")) for row in outcomes], flush=True)
    atomic_json(root / "diagnostic.json", {
        "candidate_id": original.candidate_id, "manual_control_id": patched.candidate_id,
        "source_run": target["run_path"], "model_requests": 0, "results": diagnostic,
    })
    prompts = []

    def render(context):
        incumbent = context.incumbent_source or original.proposal.source
        feedback = context.feedback if context.incumbent_source else (
            "The supplied starting kernel failed a repeated execution check.", *context.feedback
        )
        supplied = context.model_copy(update={"incumbent_source":incumbent, "feedback":feedback})
        prompt = common["tuned_prompt"](supplied) + "\nSynchronization check: " + ADVICE
        prompts.append({"sha256":hashlib.sha256(prompt.encode()).hexdigest(), "input":prompt})
        atomic_json(root / "prompts.json", prompts)
        return prompt

    config = RunConfig(**common["SETTINGS"], seed=202611089,
                       runs_dir=root/"jobs", artifacts_dir=root/"artifacts")
    with patch.object(meta, "build_prompt", render):
        summary = SearchController().run(workload, config)
    result = finalize_run(summary)
    if result.artifact_path:
        verify_artifact_evidence(result.artifact_path)
    confirmations = []
    for file in sorted(result.run_path.glob("confirmation-*.json")):
        comparisons = (read(file).get("timing") or {}).get("comparisons", [])
        strongest = min(comparisons, key=lambda c: c["summary"]["speedup"]) if comparisons else None
        confirmations.append({"baseline":strongest["name"], **strongest["summary"]} if strongest else None)
    row = {
        "requests":summary.requests, "evaluated":summary.evaluated, "outcome":result.outcome,
        "reason":result.reason, "search_score":summary.incumbent_score, "confirmation":confirmations,
        "run_path":str(result.run_path.relative_to(root)),
        "artifact_path":str(result.artifact_path.relative_to(root)) if result.artifact_path else None,
    }
    atomic_json(root/"summary.json", row)
    print(json.dumps(row), flush=True)

if __name__ == "__main__":
    main()
