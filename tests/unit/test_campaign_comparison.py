import json
from pathlib import Path
import subprocess
import sys

import pytest

from metalmind.artifacts.store import atomic_json


def _job(strategy, trial=0):
    return {"family": "bias_gelu", "case": "small", "trial": trial, "strategy": strategy}


def _row(strategy, outcome="no_win", *, trial=0, speedup=1.0, run_path=None):
    return {
        **_job(strategy, trial), "outcome": outcome, "speedup": speedup,
        "model_requests": 0 if strategy == "template" else 1, "run_path": run_path,
    }


def _evaluation(candidate, comparisons, *, passed=True):
    return {
        "event": "evaluation", "candidate_id": candidate,
        "result": {
            "status": "valid", "verification": {"passed": passed},
            "timing": {
                "domain_mode": "checked",
                "comparisons": [
                    {"name": name, "summary": {"speedup": speedup}}
                    for name, speedup in comparisons.items()
                ],
            },
        },
    }


def _run_evidence(root, name, *evaluations):
    path = root / name
    atomic_json(path / "manifest.json", {"evaluated": len(evaluations)})
    (path / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in evaluations))
    return name


def _report(tmp_path, rows, *, pending=(), output=False):
    order = [{key: row[key] for key in _job(row["strategy"])} for row in rows] + list(pending)
    path = tmp_path / "campaign"
    atomic_json(path / "campaign.json", {
        "campaign_id": "example", "jobs": len(order), "config": {"domain_mode": "checked"},
        "execution": {"job_order": order},
    })
    atomic_json(path / "summary.json", {
        "campaign_id": "example", "attempted": len(rows), "complete": not pending, "rows": rows,
    })
    args = [
        sys.executable, "-m", "scripts.report_campaign_comparison",
        str(path), "--run-root", str(tmp_path),
    ]
    output_path = tmp_path / "comparison.json"
    if output:
        args += ["--output", str(output_path)]
    result = subprocess.run(args, cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return json.loads(output_path.read_text() if output else result.stdout)


def test_strategy_denominators_include_failures_and_show_planned_jobs(tmp_path):
    report = _report(tmp_path, [
        _row("feedback", "promoted", speedup=1.44),
        _row("feedback", "failed", trial=1),
        _row("feedback", trial=2),
    ], pending=[_job("independent"), _job("feedback", 3)], output=True)

    assert (report["scope"], report["attempted"], report["planned_jobs"]) == ("partial", 3, 5)
    feedback = report["strategies"]["feedback"]
    assert (feedback["planned"], feedback["attempted"], feedback["promoted"]) == (4, 3, 1)
    assert feedback["outcomes"] == {"failed": 1, "no_win": 1, "promoted": 1}
    assert feedback["promotion_rate"] == pytest.approx(1 / 3)
    assert feedback["promotion_rate_wilson_95"] == pytest.approx([0.06149, 0.79234], abs=1e-5)
    assert feedback["all_attempted_geomean_speedup"] == pytest.approx(1.129243234657234)
    assert feedback["winner_only_geomean_speedup"] == pytest.approx(1.44)

    independent = report["strategies"]["independent"]
    assert (independent["planned"], independent["attempted"]) == (1, 0)
    assert independent["promotion_rate"] is None
    assert independent["promotion_rate_wilson_95"] is None
    assert independent["all_attempted_geomean_speedup"] is None
    assert independent["winner_only_geomean_speedup"] is None
    assert report["matched_feedback_independent"]["planned_pairs"] == 1
    assert report["matched_feedback_independent"]["attempted_pairs"] == 0


def test_pairs_use_family_case_and_trial_and_keep_failed_promotion_outcomes(tmp_path):
    report = _report(tmp_path, [
        _row("feedback", "promoted", speedup=1.25),
        _row("independent", "promoted", trial=1, speedup=1.14),
        _row("independent", "failed"),
        _row("feedback", trial=1),
        _row("feedback", trial=2),
        _row("independent", trial=2),
        _row("feedback", "promoted", trial=3, speedup=1.18),
        _row("independent", "promoted", trial=3, speedup=1.18),
        {**_row("feedback", trial=4), "case": "medium"},
        {**_row("independent", trial=4), "family": "rmsnorm"},
    ])

    assert (report["scope"], report["complete"]) == ("complete", True)
    matched = report["matched_feedback_independent"]
    assert matched["attempted_pairs"] == 4
    assert matched["promotion_outcomes"] == {
        "both_promoted": 1, "feedback_only": 1, "independent_only": 1, "neither_promoted": 1,
    }
    assert matched["pairs"][0]["independent_outcome"] == "failed"
    assert matched["pairs"][0]["search"]["status"] == "unavailable"
    assert report["strategies"]["feedback"]["promotion_rate"] == pytest.approx(2 / 5)


def test_best_search_comparison_uses_candidate_baseline_sets_not_job_unions(tmp_path):
    eager = "original_eager"
    compiled = "torch_compile_inductor_fullgraph"
    feedback = _run_evidence(
        tmp_path, "feedback",
        _evaluation("best", {eager: 2.0}),
        _evaluation("other", {eager: 1.5, compiled: 1.1}),
    )
    independent = _run_evidence(
        tmp_path, "independent", _evaluation("best", {eager: 1.7, compiled: 1.3}),
    )
    report = _report(tmp_path, [
        _row("feedback", run_path=feedback),
        _row("independent", run_path=independent),
    ])

    matched = report["matched_feedback_independent"]
    assert matched["comparable_search_pairs"] == 0
    search = matched["pairs"][0]["search"]
    assert search["status"] == "unavailable"
    assert search["reason"] == "different_verified_baseline_sets"
    assert search["feedback_best_search_speedup"] is None
    assert search["independent_best_search_speedup"] is None


def test_comparable_best_search_uses_minimum_ratio_of_verified_candidate(tmp_path):
    eager = "original_eager"
    compiled = "torch_compile_inductor_fullgraph"
    feedback = _run_evidence(
        tmp_path, "feedback",
        _evaluation("unverified", {eager: 4.0}, passed=False),
        _evaluation("best", {eager: 1.7, compiled: 1.2}),
        _evaluation("other", {eager: 2.0, compiled: 0.9}),
    )
    independent = _run_evidence(
        tmp_path, "independent", _evaluation("best", {compiled: 1.1, eager: 1.9}),
    )
    report = _report(tmp_path, [
        _row("feedback", run_path=feedback),
        _row("independent", run_path=independent),
    ])

    matched = report["matched_feedback_independent"]
    assert matched["comparable_search_pairs"] == 1
    search = matched["pairs"][0]["search"]
    assert search["status"] == "comparable"
    assert search["feedback_best_search_speedup"] == pytest.approx(1.2)
    assert search["independent_best_search_speedup"] == pytest.approx(1.1)
    assert search["comparison"] == "feedback_higher"
    assert search["feedback_baselines"] == [eager, compiled]
    assert report["strategies"]["feedback"]["winner_only_geomean_speedup"] is None


def test_ambiguous_duplicate_trials_are_rejected(tmp_path):
    from scripts.report_campaign_comparison import build_campaign_comparison

    campaign = tmp_path / "campaign"
    order = [_job("feedback"), _job("feedback"), _job("independent")]
    atomic_json(campaign / "campaign.json", {
        "campaign_id": "duplicate", "jobs": len(order), "config": {"domain_mode": "checked"},
        "execution": {"job_order": order},
    })
    atomic_json(campaign / "summary.json", {
        "campaign_id": "duplicate", "attempted": 0, "complete": False, "rows": [],
    })
    with pytest.raises(ValueError, match="duplicate"):
        build_campaign_comparison(campaign)
