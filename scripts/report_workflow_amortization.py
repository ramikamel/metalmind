"""Estimate reuse break-even from a promoted run and archived workflow timings."""
import argparse
import json
import math
from pathlib import Path
from statistics import median

from metalmind.artifacts.export import verify_artifact_evidence
from metalmind.benchmarking.statistics import summarize_pairs


def estimate_amortization(baseline_ms, candidate_ms, recorded_seconds):
    if (len(baseline_ms) != len(candidate_ms) or len(baseline_ms) < 2
            or any(not math.isfinite(v) or v <= 0 for v in (*baseline_ms, *candidate_ms))):
        raise ValueError("Expected equal-length paired positive finite latency samples")
    if not math.isfinite(recorded_seconds) or recorded_seconds < 0:
        raise ValueError("Recorded cost must be finite and nonnegative")
    saving_ms = median(a - b for a, b in zip(baseline_ms, candidate_ms))
    return {
        "paired_samples": len(baseline_ms),
        "median_paired_saving_ms": saving_ms,
        "calls_to_offset_recorded_cost": (
            math.ceil(recorded_seconds * 1000 / saving_ms) if saving_ms > 0 else None
        ),
    }


def build_report(run, artifact, workflow):
    manifest = verify_artifact_evidence(artifact)
    search = json.loads((run / "manifest.json").read_text())
    result = json.loads((run / "result.json").read_text())
    timing = json.loads(workflow.read_text())
    if (search["phase"] != "terminal" or result["outcome"] != "promoted"
            or Path(result["artifact_path"]).name != manifest.artifact_id
            or search["workload"] != manifest.workload.model_dump(mode="json")
            or search["config"]["domain_mode"] != manifest.domain_mode
            or timing["artifact_id"] != manifest.artifact_id
            or timing["domain_mode"] != manifest.domain_mode
            or timing["shape"] != list(manifest.workload.inputs[0].shape)
            or timing["outputs_match"] is not True or timing["optimized_enabled"] is not True):
        raise ValueError("Workflow must match the promoted artifact, workload, and domain")
    search_seconds = search["elapsed_seconds"]
    cooldown_seconds = search["config"]["confirmation_cooldown_seconds"]
    if any(not math.isfinite(v) or v < 0 for v in (search_seconds, cooldown_seconds)):
        raise ValueError("Invalid recorded search or cooldown duration")
    recorded_seconds = search_seconds + cooldown_seconds
    series = timing["series"]
    baseline_ms, candidate_ms = series["baseline_ms"], series["candidate_ms"]
    estimate = estimate_amortization(baseline_ms, candidate_ms, recorded_seconds)
    comparison = summarize_pairs(baseline_ms, candidate_ms).model_dump(mode="json")
    return {
        "artifact_id": manifest.artifact_id,
        "domain_mode": manifest.domain_mode,
        "shape": timing["shape"],
        "boundary": timing["boundary"],
        "workflow_comparison": comparison,
        "recorded_search_seconds": search_seconds,
        "configured_confirmation_cooldown_seconds": cooldown_seconds,
        "accounted_seconds": recorded_seconds,
        "private_verification_and_confirmation_seconds_excluding_cooldown": None,
        "first_load_seconds": None,
        "estimate_kind": "partial cost using saved median paired savings",
        **estimate,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--workflow", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_report(args.run, args.artifact, args.workflow), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
