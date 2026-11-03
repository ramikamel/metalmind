from pathlib import Path
from types import SimpleNamespace

import pytest

from metalmind import evaluation
from metalmind.artifacts.store import atomic_json
from metalmind.config import RunConfig


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    config = evaluation.CampaignConfig(
        families=("bias_gelu",),
        cases=("small",),
        strategies=("feedback", "one_shot", "independent", "template"),
        trials=1,
        run_config=RunConfig(
            runs_dir=tmp_path / "jobs",
            artifacts_dir=tmp_path / "artifacts",
            max_requests=2,
            max_candidates=2,
        ),
    )
    root = tmp_path / "campaigns"
    calls = []

    def search(self, workload, settings):
        calls.append(settings)
        run_path = settings.runs_dir / "run"
        atomic_json(run_path / "manifest.json", {"requests": settings.max_requests})
        return SimpleNamespace(run_path=run_path, requests=settings.max_requests, output_tokens=100)

    monkeypatch.setattr(evaluation.SearchController, "run", search)
    monkeypatch.setattr(evaluation, "MetaProvider", lambda config: object())
    monkeypatch.setattr(evaluation, "environment_fingerprint", lambda: "test host")
    monkeypatch.setattr(evaluation, "finalize_run", lambda summary: SimpleNamespace(
        outcome="no_win", artifact_path=None, reason="test fallback"
    ))
    monkeypatch.setattr("metalmind.runtime.worker.gpu_faulted", lambda: False)
    return config, root, calls


def test_resume_keeps_completed_jobs_and_meta_budget(campaign):
    config, root, calls = campaign
    first = evaluation.run_campaign(config, output_root=root, max_jobs=2)
    assert first["attempted"] == 2
    assert first["planned_jobs"] == 4
    assert first["max_model_requests"] == 5
    assert not first["complete"]

    finished = evaluation.run_campaign(config, output_root=root, resume=True)
    assert finished["complete"]
    assert finished["attempted"] == 4
    assert finished["rows"][:2] == first["rows"]
    assert len(calls) == 4
    assert finished["model_requests"] == 5
    template = next(row for row in finished["rows"] if row["strategy"] == "template")
    assert template["model_requests"] == 0
    assert template["proposal_slots"] == 2
    assert evaluation.run_campaign(config, output_root=root, resume=True) == finished
    assert len(calls) == 4


def test_resume_counts_interrupted_job_without_sending_again(campaign, monkeypatch):
    config, root, calls = campaign
    real_run = evaluation.SearchController.run

    def interrupt(self, workload, settings):
        real_run(self, workload, settings)
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(evaluation.SearchController, "run", interrupt)
        with pytest.raises(KeyboardInterrupt):
            evaluation.run_campaign(config, output_root=root, max_jobs=1)

    report = evaluation.run_campaign(config, output_root=root, resume=True)
    first = report["rows"][0]
    assert first["outcome"] == "failed"
    assert first["request_accounting"] == "interrupted_upper_bound"
    assert first["proposal_slots"] == calls[0].max_requests
    assert Path(first["run_path"]).is_dir()
    assert len(calls) == 4
    assert report["model_requests"] == 5


def test_resume_refuses_changed_source(campaign, monkeypatch):
    config, root, calls = campaign
    evaluation.run_campaign(config, output_root=root, max_jobs=1)
    monkeypatch.setattr(evaluation, "runtime_source_info", lambda: {"sha256": "other"})
    with pytest.raises(ValueError, match="source"):
        evaluation.run_campaign(config, output_root=root, resume=True)
    assert len(calls) == 1


def test_resume_recovers_finalized_job_after_lost_campaign_checkpoint(campaign, monkeypatch):
    config, root, calls = campaign

    def finalized_before_interrupt(summary):
        atomic_json(summary.run_path / "result.json", {
            "outcome": "no_win", "artifact_path": None, "reason": "already finalized"
        })
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(evaluation, "finalize_run", finalized_before_interrupt)
        with pytest.raises(KeyboardInterrupt):
            evaluation.run_campaign(config, output_root=root, max_jobs=1)

    report = evaluation.run_campaign(config, output_root=root, resume=True)
    assert report["rows"][0]["outcome"] == "no_win"
    assert report["rows"][0]["reason"] == "already finalized"
    assert len(calls) == 4


def test_resume_refuses_changed_environment(campaign, monkeypatch):
    config, root, calls = campaign
    evaluation.run_campaign(config, output_root=root, max_jobs=1)
    monkeypatch.setattr(evaluation, "environment_fingerprint", lambda: "new host")
    with pytest.raises(ValueError, match="environment"):
        evaluation.run_campaign(config, output_root=root, resume=True)
    assert len(calls) == 1


def test_resume_refuses_concurrent_writer(campaign):
    config, root, calls = campaign
    evaluation.run_campaign(config, output_root=root, max_jobs=1)
    folder = root / evaluation.freeze_campaign(config)["campaign_id"]
    with evaluation._campaign_lock(folder):
        with pytest.raises(ValueError, match="active writer"):
            evaluation.run_campaign(config, output_root=root, resume=True)
    assert len(calls) == 1


def test_resume_can_initialize_an_empty_campaign_directory(campaign):
    config, root, calls = campaign
    folder = root / evaluation.freeze_campaign(config)["campaign_id"]
    folder.mkdir(parents=True)
    report = evaluation.run_campaign(config, output_root=root, resume=True, max_jobs=1)
    assert report["attempted"] == len(calls) == 1
