import json
import subprocess
import sys
from metalmind.cli import main


def test_doctor_json_describes_runtime(capsys):
    exit_code = main(["doctor", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert report["schema_version"] == 1
    assert report["python_version"]
    assert exit_code == (0 if (report["capabilities"]["mps_available"] and report["capabilities"]["compile_shader"]) else 1)


def test_doctor_smoke_without_torch_still_emits_diagnostic_json():
    result = subprocess.run([sys.executable, "-c",
        "import sys; sys.modules['torch'] = None; "
        "from metalmind.cli import main; "
        "raise SystemExit(main(['doctor', '--json', '--smoke']))"],
        capture_output=True, text=True, check=False)
    report = json.loads(result.stdout)
    assert result.returncode == 1
    assert report["torch_version"] is None
    assert report["shader_health_passed"] is False
    assert report["unavailable_reason"]


def test_doctor_rejects_missing_shader_api(monkeypatch, capsys):
    import torch
    monkeypatch.setattr(torch.mps, "compile_shader", None)
    assert main(["doctor", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["capabilities"]["compile_shader"] is False
    assert report["unavailable_reason"]


def test_analyze_bundled_workload_and_campaign_preview(capsys):
    assert main(["analyze","--workload","bias_gelu","--case","small","--json"])==0
    report=json.loads(capsys.readouterr().out)
    assert report['workload']['family']=='bias_gelu'
    assert main(["evaluate","--suite","configs/evaluation.yaml"])==0
    campaign=json.loads(capsys.readouterr().out)
    assert campaign['jobs']==108


def test_module_commands_are_offline_and_keep_no_artifact_fallback(capsys, tmp_path, monkeypatch):
    from metalmind.agent.meta import MetaProvider
    def forbidden(*args): raise AssertionError("Module reuse must not make model requests")
    monkeypatch.setattr(MetaProvider,"propose",forbidden)
    assert main(["analyze-module","--factory","examples.module_workloads:make_cpu_demo"]) == 0
    analysis = json.loads(capsys.readouterr().out)
    assert analysis["supported"] and analysis["regions"]
    assert main(["optimize-module","--factory","examples.module_workloads:make_cpu_demo","--output-dir",str(tmp_path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["outcome"] == "no_artifact" and result["model_requests"] == 0


def test_campaign_execution_flags_require_explicit_execute(capsys):
    for flags in (["--resume"], ["--max-jobs", "2"]):
        assert main(["evaluate", "--suite", "configs/evaluation.yaml", *flags]) == 1
        assert "require --execute" in json.loads(capsys.readouterr().out)["reason"]


def test_doctor_holds_gpu_lock_until_fault_is_cleared(monkeypatch, capsys):
    from contextlib import contextmanager
    from dataclasses import replace
    from metalmind import cli

    info = cli.probe_environment()
    available = replace(info.capabilities, mps_available=True, compile_shader=True)
    monkeypatch.setattr(cli, "probe_environment", lambda: replace(info, capabilities=available))
    events = []

    @contextmanager
    def lock():
        events.append("lock")
        try:
            yield
        finally:
            events.append("unlock")

    monkeypatch.setattr("metalmind.runtime.lock.gpu_lock", lock)
    monkeypatch.setattr("metalmind.runtime.health.check_health", lambda: events.append("smoke") or True)
    monkeypatch.setattr("metalmind.runtime.worker.clear_device_fault", lambda: events.append("clear"))
    assert main(["doctor", "--json", "--smoke"]) == 0
    assert json.loads(capsys.readouterr().out)["shader_health_passed"]
    assert events == ["lock", "smoke", "clear", "unlock"]
