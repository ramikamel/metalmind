from pathlib import Path

import pytest

from metalmind.artifacts.store import atomic_json
from scripts import archive_campaign


@pytest.fixture
def finished_campaign(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    campaign = root / "campaign"
    job = {"family": "bias_gelu", "case": "small", "strategy": "feedback", "trial": 0}
    path = Path("runs/one")
    workload = {"family": "bias_gelu", "workload_id": "bias_gelu:small"}
    atomic_json(campaign / "campaign.json", {
        "campaign_id": "test", "jobs": 1, "config": {"domain_mode": "checked", "run_config": {"max_requests": 2}},
        "workloads": [{"family": "bias_gelu", "case": "small", "workload": workload}],
        "execution": {"source_sha256": "source", "job_order": [job], "max_model_requests": 2},
    })
    summary = {
        "campaign_id": "test", "attempted": 1, "complete": True, "model_requests": 2,
        "rows": [{**job, "run_path": str(path), "artifact_path": None, "outcome": "no_win",
                  "speedup": 1.0, "model_requests": 2, "requests": 2, "proposal_slots": 2,
                  "request_accounting": "search_ledger", "reason": "no improvement"}],
    }
    atomic_json(campaign / "summary.json", summary)
    atomic_json(root / path / "manifest.json", {
        "evaluated": 0, "requests": 2, "workload": workload,
        "config": {"domain_mode": "checked", "max_requests": 2},
    })
    atomic_json(root / path / "result.json", {
        "outcome": "no_win", "artifact_path": None, "reason": "no improvement",
    })
    monkeypatch.setattr(archive_campaign, "_verified_source_commit", lambda *_: "revision")
    return root, campaign, summary


def test_archive_preserves_raw_data_and_reproduces_summary(finished_campaign, tmp_path):
    import hashlib
    import json

    root, campaign, _ = finished_campaign
    destination = tmp_path / "archive"
    archive_campaign.archive_completed_campaign(campaign, root, destination)
    assert (destination / "raw/runs/one/manifest.json").read_bytes() == (
        root / "runs/one/manifest.json"
    ).read_bytes()
    report = json.loads((destination / "analysis.json").read_text())
    assert report["complete"] is True and report["strategies"]["feedback"]["model_requests"] == 2
    assert json.loads((destination / "comparison.json").read_text())["scope"] == "complete"
    checksums = json.loads((destination / "sha256.json").read_text())
    for name, expected in checksums.items():
        assert hashlib.sha256((destination / name).read_bytes()).hexdigest() == expected


@pytest.mark.parametrize("case", ["incomplete", "path_escape", "wrong_request_total"])
def test_archive_rejects_incomplete_unsafe_or_inconsistent_ledger(finished_campaign, tmp_path, case):
    root, campaign, summary = finished_campaign
    if case == "incomplete":
        summary["complete"] = False
    elif case == "path_escape":
        summary["rows"][0]["run_path"] = "../outside"
    else:
        summary["model_requests"] = 1
    atomic_json(campaign / "summary.json", summary)
    dest = tmp_path / "archive"
    with pytest.raises(ValueError):
        archive_campaign.archive_completed_campaign(campaign, root, dest)
    assert not dest.exists()


@pytest.mark.parametrize("field,value", [
    ("speedup", 1.3), ("outcome", "failed"), ("proposal_slots", 1),
])
def test_archive_checks_summary_against_copied_run(finished_campaign, tmp_path, field, value):
    root, campaign, summary = finished_campaign
    summary["rows"][0][field] = value
    atomic_json(campaign / "summary.json", summary)
    with pytest.raises(ValueError):
        archive_campaign.archive_completed_campaign(campaign, root, tmp_path / "archive")
