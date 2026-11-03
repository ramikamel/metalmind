"""Summarize recorded campaign rows without executing kernels or calling a model."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics

from metalmind.artifacts.store import atomic_json
from metalmind.reporting.summary import build_run_summary


THRESHOLDS = (1.0, 1.05, 1.2)


def _read(path):
    return json.loads(path.read_text())


def _job_evidence(row, run_root, domain_mode):
    path = row["run_path"]
    if path is None:
        return {**row, "evaluations_started": 0, "statuses": {}, "responses": 0,
                "schema_rejections": 0, "reported_tokens": {}, "scores": [],
                "correct": 0, "search_seconds": None, "baselines": []}
    bundle = build_run_summary(run_root / path)
    proposals = [e for e in bundle["events"] if e["event"] == "model_response" and row["strategy"] != "template"]
    tokens = Counter()
    for event in proposals:
        for name in ("input_tokens", "output_tokens", "reasoning_tokens"):
            value = event["usage"].get(name)
            if value is not None:
                tokens[name] += value
    scores, baselines = [], set()
    correct = 0
    for event in bundle["events"]:
        if event["event"] != "evaluation":
            continue
        result = event["result"]
        correct += bool((result.get("verification") or {}).get("passed"))
        timing = result.get("timing")
        if timing:
            if timing["domain_mode"] != domain_mode:
                raise ValueError("Search timing domain differs from the frozen campaign")
            comparisons = timing["comparisons"]
            baselines.update(c["name"] for c in comparisons)
            scores.append(min(c["summary"]["speedup"] for c in comparisons))
    return {**row, "evaluations_started": bundle["manifest"].get("evaluated", 0),
            "statuses": bundle["candidate_counts"], "responses": len(proposals),
            "schema_rejections": sum(e["error"] is not None for e in proposals),
            "reported_tokens": dict(tokens), "scores": scores, "correct": correct,
            "search_seconds": bundle["manifest"].get("elapsed_seconds"),
            "baselines": sorted(baselines)}


def _strategy_metrics(jobs):
    statuses = Counter()
    for job in jobs:
        statuses.update(job["statuses"])
    scores = [score for job in jobs for score in job["scores"]]
    durations = [job["search_seconds"] for job in jobs if job["search_seconds"] is not None]
    return {
        "attempted": len(jobs),
        "outcomes": dict(Counter(job["outcome"] for job in jobs)),
        "model_requests": sum(job["model_requests"] for job in jobs),
        "response_count": sum(job["responses"] for job in jobs),
        "schema_rejections": sum(job["schema_rejections"] for job in jobs),
        "reported_input_tokens": sum(job["reported_tokens"].get("input_tokens", 0) for job in jobs),
        "reported_output_tokens": sum(job["reported_tokens"].get("output_tokens", 0) for job in jobs),
        "evaluations_started": sum(job["evaluations_started"] for job in jobs),
        "candidate_statuses": dict(statuses),
        "verified_correct_candidates": sum(job["correct"] for job in jobs),
        "valid_timing_candidates": len(scores),
        "search_threshold_candidates": {f"{x:.2f}": sum(score >= x for score in scores) for x in THRESHOLDS},
        "median_search_seconds": statistics.median(durations) if durations else None,
        "geometric_mean_final_speedup": math.exp(statistics.mean(math.log(job["speedup"]) for job in jobs)),
    }


def build_campaign_report(campaign_dir, *, run_root=Path(".")):
    campaign_dir = Path(campaign_dir)
    frozen = _read(campaign_dir / "campaign.json")
    summary = _read(campaign_dir / "summary.json")
    order = frozen["execution"]["job_order"]
    rows = summary["rows"]
    if (summary["campaign_id"] != frozen["campaign_id"] or summary["attempted"] != len(rows)
        or len(rows) > len(order) or frozen["jobs"] != len(order)
        or summary["complete"] != (len(rows) == len(order))
        or any({key: row[key] for key in job} != job for row, job in zip(rows, order))):
        raise ValueError("Campaign summary does not match the frozen order")
    mode = frozen["config"]["domain_mode"]
    jobs = [_job_evidence(row, Path(run_root), mode) for row in rows]
    groups = {strategy: [job for job in jobs if job["strategy"] == strategy]
              for strategy in sorted({job["strategy"] for job in jobs})}
    independent = {(job["family"], job["case"], job["trial"]): job
                   for job in groups.get("independent", [])}
    matched = []
    for job in groups.get("feedback", []):
        key = (job["family"], job["case"], job["trial"])
        other = independent.get(key)
        if other:
            matched.append({
                "family": key[0], "case": key[1], "trial": key[2],
                "feedback_outcome": job["outcome"], "independent_outcome": other["outcome"],
                "feedback_best_search_speedup": max(job["scores"], default=None),
                "independent_best_search_speedup": max(other["scores"], default=None),
            })
    return {
        "campaign_id": frozen["campaign_id"], "domain_mode": mode,
        "planned_jobs": frozen["jobs"], "attempted": len(jobs), "complete": summary["complete"],
        "strategies": {strategy: _strategy_metrics(group) for strategy, group in groups.items()},
        "matched_feedback_independent": matched, "jobs": jobs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument("--run-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_campaign_report(args.campaign_dir, run_root=args.run_root)
    if args.output:
        atomic_json(args.output, report)
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
