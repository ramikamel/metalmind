from pathlib import Path
import pytest
import torch
from metalmind.demonstration.checks import check_artifact_reuse

pytestmark=[pytest.mark.gpu,pytest.mark.skipif(not torch.backends.mps.is_available(),reason="MPS unavailable")]


def test_demo_fresh_check_uses_existing_artifact_and_correct_fallbacks():
    path=Path(__file__).resolve().parents[2]/"results/2026-11-02-fp32-pilot/artifacts/db066f6dfe2e74cefc0c335852229dd2ed6a9058518279eabb7b3a28342caa87"
    result=check_artifact_reuse(path)
    if result["outcome"]=="unavailable": pytest.skip(result["reason"])
    assert result["outcome"]=="passed" and result["new_values_match"]
    assert result["changed_shape_fallback"] and result["gradient_fallback"]
    assert result["new_performance_measurement"] is False
