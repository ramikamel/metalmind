from metalmind.artifacts.store import ArtifactStore
from metalmind.reporting.summary import build_run_summary
from metalmind.reporting.html import render_report


def test_report_includes_failures_and_escapes_model_text(tmp_path):
    store=ArtifactStore(tmp_path)
    store.write_manifest({"phase":"search_complete","requests":2,"evaluated":1})
    store.append_event({"event":"evaluation","candidate_id":"a","result":{"status":"compile_error","reason":"<script>alert(1)</script>"}})
    summary=build_run_summary(tmp_path)
    assert summary['candidate_counts']['compile_error']==1
    path=render_report(summary,tmp_path/'report.html')
    text=path.read_text()
    assert '<script>alert(1)</script>' not in text
    assert '&lt;script&gt;' in text


def test_report_includes_saved_private_and_confirmation_evidence(tmp_path):
    from metalmind.artifacts.store import atomic_json
    store = ArtifactStore(tmp_path)
    store.write_manifest({"phase": "terminal", "archive": [{"score": 1.2}]})
    atomic_json(tmp_path / "result.json", {"outcome": "promoted"})
    atomic_json(tmp_path / "private-verification.json", {"status": "valid", "verification": {"passed": True}})
    for i in (1, 2):
        atomic_json(tmp_path / f"confirmation-{i}.json", {"status": "valid", "timing": {"comparisons": [{"name": "original_eager"}]}})
    summary = build_run_summary(tmp_path)
    assert summary["final_evidence"]["private_verification"]["verification"]["passed"]
    assert len(summary["final_evidence"]["confirmations"]) == 2
    assert summary["stages"]["promoted"] is True
    assert summary["stages"]["final_verified"] is True
