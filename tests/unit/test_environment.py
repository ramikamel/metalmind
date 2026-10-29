import json
from dataclasses import asdict

from metalmind.runtime.environment import probe_environment


def test_report_is_serializable_and_contains_actual_runtime():
    info = probe_environment()
    report = asdict(info)
    assert report["schema_version"] == 1
    assert report["python_version"]
    assert report["torch_version"]
    assert isinstance(report["capabilities"]["mps_available"], bool)
    json.dumps(report, allow_nan=False)


def test_unavailable_device_has_actionable_reason(monkeypatch):
    import torch
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    info = probe_environment()
    assert not info.capabilities.mps_available
    assert info.unavailable_reason
