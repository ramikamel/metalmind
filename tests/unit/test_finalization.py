from metalmind.api import OptimizationResult
from metalmind.artifacts.export import finalize_run
from metalmind.agent.controller import RunSummary
from metalmind.config import RunConfig


def test_unmeasured_run_returns_fallback_without_export(sample_workload,tmp_path):
    summary=RunSummary(workload=sample_workload,config=RunConfig(),run_path=tmp_path,outcome="budget_exhausted")
    result=finalize_run(summary)
    assert result.artifact_path is None
    assert result.outcome=="budget_exhausted"
    fn=result.build_callable(fallback=lambda x:x+1)
    assert fn(4)==5


def test_revalidation_preserves_reproducibility_metadata(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace
    import metalmind.artifacts.loader as loader
    import metalmind.artifacts.export as export
    from metalmind.agent.candidates import make_candidate
    from metalmind.evaluation import template_proposals
    from metalmind.workloads.registry import get_workload, resolve_reference
    workload = get_workload("bias_gelu", "small")
    candidate = make_candidate(template_proposals(workload)[0].proposal, workload)
    manifest = SimpleNamespace(workload=workload, candidate=candidate, domain_mode="checked")
    monkeypatch.setattr(loader, "verify_artifact_evidence", lambda p: manifest)
    monkeypatch.setattr(export, "finalize_run", lambda summary: summary)
    summary = loader.revalidate_artifact(tmp_path / "old", fallback=resolve_reference("bias_gelu"), config=RunConfig(runs_dir=tmp_path / "runs"))
    saved = json.loads((summary.run_path / "manifest.json").read_text())
    assert saved["config"]["model"]
    assert saved["environment"]["python_version"]
    assert saved["workload"]["reference_hash"] == workload.reference_hash
    assert (summary.run_path / "candidates" / (candidate.candidate_id + ".metal")).read_text() == candidate.proposal.source
