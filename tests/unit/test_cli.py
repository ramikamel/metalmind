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
