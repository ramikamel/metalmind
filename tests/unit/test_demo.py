from pathlib import Path
import json
import shutil
import urllib.request
import pytest
import torch
from metalmind.demonstration.evidence import load_demo_evidence
from metalmind.demonstration.story import build_chapters, safe_text

ROOT=Path(__file__).resolve().parents[2]/"results"
PILOT="2026-11-02-fp32-pilot"
MODULE="2026-11-02-module-optimizer"

@pytest.fixture
def evidence(tmp_path):
    root=tmp_path/"evidence";root.mkdir()
    for name in (PILOT,MODULE): shutil.copytree(ROOT/name,root/name)
    compiled=ROOT/"2026-11-02-module-baselines"
    if compiled.exists(): shutil.copytree(compiled,root/compiled.name)
    return root


def test_replay_validates_all_outcomes_without_network_or_gpu(evidence,monkeypatch):
    def forbidden(*a,**kw): raise AssertionError("Replay must stay offline and CPU-only")
    monkeypatch.setattr(urllib.request.OpenerDirector,"open",forbidden)
    monkeypatch.setattr(torch.mps,"compile_shader",forbidden)
    data=load_demo_evidence(evidence)
    assert data["mode"]=="saved_evidence_replay" and data["model_requests"]==0
    assert data["pilot"]["jobs"]==9 and data["pilot"]["promoted"]==2 and data["pilot"]["no_win"]==7
    assert data["module"]["private_cases"]==8
    assert len(data["function_winners"])==2
    chapters=build_chapters(data)
    assert any("7 no-win" in " ".join(c["lines"]) for c in chapters)
    assert all(c["label"] for c in chapters)


def test_changed_kernel_is_rejected(evidence):
    path=next((evidence/PILOT/"artifacts").glob("*/kernel.metal"))
    path.write_text(path.read_text()+"\n// edited")
    with pytest.raises(ValueError): load_demo_evidence(evidence)


def test_fabricated_module_summary_is_rejected_from_raw_pairs(evidence):
    release=json.loads((evidence/MODULE/"release.json").read_text());folder=evidence/MODULE/release["primary_run"]
    result=json.loads((folder/"result.json").read_text())
    result["confirmations"][0]["summary"]["speedup"]=99.
    (folder/"result.json").write_text(json.dumps(result))
    report=json.loads((folder/"confirmation-1.json").read_text());report["summary"]["speedup"]=99.
    (folder/"confirmation-1.json").write_text(json.dumps(report))
    with pytest.raises(ValueError,match="raw|summary"): load_demo_evidence(evidence)


def test_run_identifier_cannot_escape_evidence_directory(evidence):
    path=evidence/MODULE/"release.json";data=json.loads(path.read_text());data["primary_run"]="../../.env";path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="identifier"): load_demo_evidence(evidence)


def test_symlink_cannot_read_outside_evidence_root(evidence,tmp_path):
    kernel=next((evidence/PILOT/"artifacts").glob("*/kernel.metal"))
    outside=tmp_path/"outside.metal";outside.write_text(kernel.read_text());kernel.unlink();kernel.symlink_to(outside)
    with pytest.raises(ValueError,match="outside"): load_demo_evidence(evidence)


def test_terminal_text_cannot_emit_escape_or_bidi_controls():
    assert "\x1b" not in safe_text("source\x1b]52;c;payload\x07")
    assert "\u202e" not in safe_text("name\u202e")
    assert safe_text("a\nb\tc")=="a\nb\tc"


def test_cli_demo_exports_validated_replay(evidence,tmp_path,capsys,monkeypatch):
    from metalmind.cli import main
    from metalmind.agent.meta import MetaProvider
    def forbidden(*a,**kw): raise AssertionError("No generation during demo")
    monkeypatch.setattr(MetaProvider,"propose",forbidden)
    output=tmp_path/"demo.json"
    assert main(["demo","--evidence-root",str(evidence),"--json","--output",str(output)])==0
    result=json.loads(capsys.readouterr().out)
    assert result==json.loads(output.read_text())
    assert result["model_requests"]==0 and result["chapters"]


def test_compiler_counterexample_and_disabled_gate_are_preserved(evidence):
    data=load_demo_evidence(evidence)
    assert data["compiled_module_comparison"]["outcome"]=="no_confirmed_gain"
    compiled=[c for c in data["compiled_module_comparison"]["comparisons"] if "inductor" in c["baseline"]]
    assert len(compiled)==2 and all(c["speedup"]<1 for c in compiled)
    assert data["module_gate"]["enabled"] is False
    assert data["module_gate"]["outcome"]=="no_win"


def test_fabricated_compiler_outcome_is_rejected(evidence):
    path=evidence/"2026-11-02-module-baselines/report.json"
    data=json.loads(path.read_text());data["compiled_comparison_outcome"]="confirmed_gain";path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="outcome"): load_demo_evidence(evidence)


def test_fresh_check_never_imports_a_custom_reference(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from metalmind.demonstration.checks import check_artifact_reuse
    import metalmind.artifacts.export as export
    import metalmind.workloads.registry as registry
    import metalmind.runtime.lock as lock
    import metalmind.runtime.worker as worker
    monkeypatch.setattr(torch.backends.mps,"is_available",lambda:True)
    monkeypatch.setattr(worker,"gpu_faulted",lambda:False)
    monkeypatch.setattr(lock,"gpu_lock",nullcontext)
    manifest=SimpleNamespace(workload=SimpleNamespace(reference_id="python:untrusted:run",family="bias_gelu"))
    monkeypatch.setattr(export,"verify_artifact_evidence",lambda p:manifest)
    def forbidden(*a): raise AssertionError("Evidence must not trigger host imports")
    monkeypatch.setattr(registry,"resolve_reference",forbidden)
    result=check_artifact_reuse(Path("unused"))
    assert result["outcome"]=="unavailable" and "registered" in result["reason"]
