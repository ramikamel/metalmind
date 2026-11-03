from pathlib import Path
import pytest
import torch
from examples.module_workloads import make_demo
from metalmind.graph.capture import capture_module
from metalmind.graph.rewrite import ModuleOptimizationConfig,GuardedModule,_select_artifacts,_rewrite_graph,_verify_private,optimize_module
from metalmind.runtime.lock import gpu_lock

pytestmark = [pytest.mark.gpu,pytest.mark.skipif(not torch.backends.mps.is_available(),reason="MPS unavailable")]
ARTIFACT = Path(__file__).resolve().parents[2] / "results/2026-11-02-fp32-pilot/artifacts/db066f6dfe2e74cefc0c335852229dd2ed6a9058518279eabb7b3a28342caa87"


def test_assumed_artifact_requires_module_opt_in():
    module,inputs = make_demo()
    result = optimize_module(module,inputs,config=ModuleOptimizationConfig(artifact_paths=(ARTIFACT,)))
    assert not result.enabled and result.outcome == "no_artifact"
    assert any("opt-in" in e for e in result.analysis.exclusions)


def test_real_promoted_region_preserves_private_module_outputs_and_fallbacks():
    module,inputs = make_demo()
    captured = capture_module(module,inputs)
    config = ModuleOptimizationConfig(artifact_paths=(ARTIFACT,),domain_mode="assume")
    with gpu_lock():
        selected,identities,issues = _select_artifacts(captured,config)
        if not selected: pytest.skip("Recorded artifact requires its exact GPU/runtime environment: " + str(issues))
        assert len(identities) == 1
        candidate = GuardedModule(module,_rewrite_graph(captured,selected),captured.guard)
        verification = _verify_private(captured,candidate,selected,config,tuple(range(8)))
        assert verification["passed"] and verification["cases"] == 8
        with torch.inference_mode():
            module.bias.add_(.1)
            torch.testing.assert_close(candidate(*inputs),module(*inputs),atol=1e-5,rtol=1e-4)
            assert candidate.last_path == "optimized"
            changed = inputs[0][:2].contiguous()
            torch.testing.assert_close(candidate(changed),module(changed))
            assert candidate.last_path == "fallback"
        x = inputs[0].detach().requires_grad_(True)
        candidate(x).sum().backward()
        assert x.grad is not None and module.bias.grad is not None
        assert candidate.last_path == "fallback"


def test_example_domain_is_verified_even_with_assumed_mode(monkeypatch):
    from metalmind.artifacts.export import verify_artifact_evidence,environment_fingerprint
    if verify_artifact_evidence(ARTIFACT).environment_fingerprint != environment_fingerprint():
        pytest.skip("Recorded artifact requires its exact GPU/runtime environment")
    import metalmind.graph.rewrite as rewrite
    monkeypatch.setattr(rewrite,"_confirm",lambda *a,**kw:("no_win",[]))
    module,inputs = make_demo()
    inputs[0].fill_(100)
    result = optimize_module(module,inputs,config=ModuleOptimizationConfig(artifact_paths=(ARTIFACT,),domain_mode="assume"))
    assert result.outcome == "failed" and not result.enabled
    assert "Example" in result.reason


def test_wrong_compiled_module_is_excluded_before_timing(monkeypatch):
    from metalmind.graph.baselines import collect_module_baselines
    module,inputs=make_demo()
    before=module.bias._version
    monkeypatch.setattr(torch,"compile",lambda *a,**kw:lambda x:torch.zeros_like(x))
    baselines,unavailable=collect_module_baselines(module,inputs,tuple(range(8)))
    assert [name for name,_ in baselines]==["original_eager_whole_module"]
    assert unavailable and "torch_compile" in unavailable[0]
    assert module.bias._version==before
