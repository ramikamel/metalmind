import pytest
import torch
from examples.module_workloads import ResidualActivation
from metalmind.graph.capture import capture_module
from metalmind.graph.profile import analyze_module
from metalmind.workloads.registry import resolve_reference

pytestmark = [pytest.mark.gpu,pytest.mark.skipif(not torch.backends.mps.is_available(),reason="MPS unavailable")]


def test_mps_region_extraction_matches_registered_semantics_and_keeps_parameters_live():
    module = ResidualActivation(257).eval().to("mps")
    x = torch.randn(8,257,device="mps")
    captured = capture_module(module,(x,))
    region = captured.regions[0]
    assert region.info.workload is not None
    with torch.inference_mode():
        torch.testing.assert_close(region.function(*region.example_inputs),resolve_reference("bias_gelu")(*region.example_inputs),atol=0,rtol=0)
        module.bias.add_(.1)
        assert captured.guard.matches((x,))
        torch.testing.assert_close(captured.graph(x),module(x),atol=0,rtol=0)


def test_module_profile_reports_standalone_measurements_or_explicit_drift_rejection():
    module = ResidualActivation(257).eval().to("mps")
    x = torch.randn(8,257,device="mps")
    analysis = analyze_module(module,(x,),measure=True)
    assert analysis.supported and len(analysis.regions) == 1
    if analysis.whole_latency:
        assert len(analysis.whole_latency.samples_ms) == 10
        assert analysis.whole_latency.median_ms > 0
    else:
        assert any("drift" in e for e in analysis.exclusions)
    if analysis.regions[0].latency:
        assert "not exclusive" in analysis.ranking_basis
