"""Compare recorded campaign strategies and matched trials offline."""
import argparse
from collections import Counter
import json
from math import sqrt
from pathlib import Path
from statistics import geometric_mean

from metalmind.artifacts.store import ArtifactStore, atomic_json
from scripts.report_campaign import build_campaign_report


PAIR_FIELDS = ("family", "case", "trial")
PROMOTION_OUTCOMES = {
    (True, True): "both_promoted",
    (True, False): "feedback_only",
    (False, True): "independent_only",
    (False, False): "neither_promoted",
}


def _wilson_interval(rate, count):
    z = 1.959963984540054
    adjustment = z ** 2 / count
    center = (rate + adjustment / 2) / (1 + adjustment)
    margin = z * sqrt((rate * (1 - rate) + adjustment / 4) / count) / (1 + adjustment)
    return [center - margin if rate else 0.0, center + margin if rate < 1.0 else 1.0]


def _strategy_comparison(jobs, planned, metrics):
    attempted = len(jobs)
    winners = [job["speedup"] for job in jobs if job["outcome"] == "promoted"]
    promoted = len(winners)
    rate = promoted / attempted if attempted else None
    all_speedups = winners + [1.0] * (attempted - promoted)
    return {
        "planned": planned,
        "attempted": attempted,
        "unattempted": planned - attempted,
        "outcomes": metrics.get("outcomes", {}),
        "model_requests": metrics.get("model_requests", 0),
        "evaluations_started": metrics.get("evaluations_started", 0),
        "promoted": promoted,
        "promotion_rate": rate,
        "promotion_rate_wilson_95": _wilson_interval(rate, attempted) if attempted else None,
        "all_attempted_geomean_speedup": geometric_mean(all_speedups) if all_speedups else None,
        "winner_only_geomean_speedup": geometric_mean(winners) if winners else None,
    }


def _pair_key(job):
    return tuple(job[field] for field in PAIR_FIELDS)


def _best_search(job, run_root):
    if job["run_path"] is None:
        return None
    best = None
    for event in ArtifactStore(run_root / job["run_path"]).read_events():
        if event["event"] != "evaluation":
            continue
        result = event["result"]
        timing = result.get("timing")
        if (result["status"] != "valid" or not (result.get("verification") or {}).get("passed")
                or not timing or not timing["comparisons"]):
            continue
        comparisons = timing["comparisons"]
        score = min(item["summary"]["speedup"] for item in comparisons)
        # Keep the first tied candidate, as the search archive does.
        if best is None or score > best["score"]:
            best = {
                "candidate_id": event.get("candidate_id"),
                "score": score,
                "baselines": sorted({item["name"] for item in comparisons}),
            }
    return best


def _search_comparison(feedback, independent, run_root):
    left = _best_search(feedback, run_root)
    right = _best_search(independent, run_root)
    search = {
        "status": "unavailable",
        "feedback_candidate_id": left["candidate_id"] if left else None,
        "independent_candidate_id": right["candidate_id"] if right else None,
        "feedback_baselines": left["baselines"] if left else None,
        "independent_baselines": right["baselines"] if right else None,
        "feedback_best_search_speedup": None,
        "independent_best_search_speedup": None,
        "comparison": None,
    }
    if left is None or right is None:
        return {**search, "reason": "missing_verified_search_timing"}
    if left["baselines"] != right["baselines"]:
        return {**search, "reason": "different_verified_baseline_sets"}
    feedback_score, independent_score = left["score"], right["score"]
    comparison = "equal"
    if feedback_score > independent_score:
        comparison = "feedback_higher"
    elif independent_score > feedback_score:
        comparison = "independent_higher"
    return {
        **search,
        "status": "comparable",
        "feedback_best_search_speedup": feedback_score,
        "independent_best_search_speedup": independent_score,
        "comparison": comparison,
    }


def _matched_comparison(report, order, run_root):
    planned = {}
    for strategy in ("feedback", "independent"):
        keys = [_pair_key(job) for job in order if job["strategy"] == strategy]
        planned[strategy] = set(keys)
        if len(planned[strategy]) != len(keys):
            raise ValueError(f"Cannot match duplicate {strategy} trials")
    jobs = {(job["strategy"], _pair_key(job)): job for job in report["jobs"]}
    promotion_counts = dict.fromkeys(PROMOTION_OUTCOMES.values(), 0)
    pairs = []
    for match in report["matched_feedback_independent"]:
        key = _pair_key(match)
        left, right = jobs["feedback", key], jobs["independent", key]
        outcomes = (left["outcome"], right["outcome"])
        promotion = PROMOTION_OUTCOMES[tuple(outcome == "promoted" for outcome in outcomes)]
        promotion_counts[promotion] += 1
        pairs.append({
            **dict(zip(PAIR_FIELDS, key)),
            "feedback_outcome": outcomes[0],
            "independent_outcome": outcomes[1],
            "promotion_outcome": promotion,
            "search": _search_comparison(left, right, run_root),
        })
    return {
        "planned_pairs": len(planned["feedback"] & planned["independent"]),
        "attempted_pairs": len(pairs),
        "promotion_outcomes": promotion_counts,
        "comparable_search_pairs": sum(pair["search"]["status"] == "comparable" for pair in pairs),
        "pairs": pairs,
    }


def build_campaign_comparison(campaign_dir, *, run_root=Path(".")):
    campaign_dir, run_root = Path(campaign_dir), Path(run_root)
    report = build_campaign_report(campaign_dir, run_root=run_root)
    order = json.loads((campaign_dir / "campaign.json").read_text())["execution"]["job_order"]
    planned = Counter(job["strategy"] for job in order)
    strategies = {
        strategy: _strategy_comparison(
            [job for job in report["jobs"] if job["strategy"] == strategy],
            planned[strategy], report["strategies"].get(strategy, {}),
        )
        for strategy in sorted(planned)
    }
    return {
        "campaign_id": report["campaign_id"],
        "domain_mode": report["domain_mode"],
        "scope": "complete" if report["complete"] else "partial",
        "complete": report["complete"],
        "attempted": report["attempted"],
        "planned_jobs": report["planned_jobs"],
        "interpretation": (
            "The recorded jobs support descriptive comparisons only. A partial scope covers the "
            "frozen prefix. All attempted jobs enter the promotion denominator. Non-promoted jobs "
            "contribute 1.0 to the all-job final speedup, and unattempted jobs are excluded. "
            "Winner speedups use the recorded minimum ratio across confirmation comparisons. "
            "The Wilson intervals describe promotion rates under a binomial model."
        ),
        "strategies": strategies,
        "matched_feedback_independent": _matched_comparison(report, order, run_root),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument("--run-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_campaign_comparison(args.campaign_dir, run_root=args.run_root)
    if args.output:
        atomic_json(args.output, report)
    else:
        print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
