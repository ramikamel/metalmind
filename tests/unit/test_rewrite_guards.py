import torch
from test_region_partition import Block
from metalmind.graph.capture import capture_module
from metalmind.graph.rewrite import optimize_module, ModuleOptimizationConfig


def test_graph_guard_accepts_live_parameter_values_but_rejects_changed_state():
    module = Block().eval()
    x = torch.ones(2, 3)
    capture = capture_module(module, (x,))
    with torch.inference_mode():
        assert capture.guard.matches((x,))
        module.bias.add_(1)
        assert capture.guard.matches((x,))
        module.bias = torch.nn.Parameter(torch.zeros(3))
        assert not capture.guard.matches((x,))


def test_missing_artifacts_falls_back_with_original_output_and_gradients():
    module = Block().eval()
    x = torch.ones(2, 3, requires_grad=True)
    result = optimize_module(module, (x,), config=ModuleOptimizationConfig())
    assert not result.enabled and result.outcome == "no_artifact"
    actual = result.module(x)
    torch.testing.assert_close(actual, module(x))
    actual["activation"].sum().backward()
    assert x.grad is not None


class ScaledBlock(Block):
    def __init__(self):
        super().__init__()
        self.scale = .5
    def forward(self, x):
        return x @ self.weight + self.bias * self.scale


def test_hooks_shapes_autocast_scalars_and_code_changes_disable_guard(monkeypatch):
    module = ScaledBlock().eval()
    x = torch.ones(2, 3)
    guard = capture_module(module, (x,)).guard
    with torch.inference_mode():
        assert not guard.matches((torch.ones(4, 3),))
        with torch.autocast("cpu"):
            assert not guard.matches((x,))
        module.scale = .25
        assert not guard.matches((x,))
        module.scale = .5
        hook = module.register_forward_hook(lambda *args: None)
        assert not guard.matches((x,))
        hook.remove()
        assert guard.matches((x,))
        monkeypatch.setattr(module.forward.__func__, "__code__", Block.forward.__code__)
        assert not guard.matches((x,))



def test_dependent_rewrites_keep_outputs_and_live_parameters():
    from test_region_partition import TwoRegions
    from metalmind.graph.rewrite import _rewrite_graph, GuardedModule
    from metalmind.workloads.registry import resolve_reference
    module = TwoRegions().eval()
    x = torch.randn(2, 3)
    captured = capture_module(module, (x,))
    replacements = {r.info.region_id:resolve_reference(r.info.family) for r in captured.regions}
    rewritten = _rewrite_graph(captured, replacements)
    guarded = GuardedModule(module, rewritten, captured.guard)
    with torch.inference_mode():
        torch.testing.assert_close(guarded(x), module(x))
        module.bias.add_(.3)
        torch.testing.assert_close(guarded(x), module(x))
        module.bias = torch.nn.Parameter(torch.ones(3))
        torch.testing.assert_close(guarded(x), module(x))
    assert guarded.last_path == "fallback"


def test_integrated_gate_rejects_second_session_loss(monkeypatch):
    import metalmind.graph.rewrite as rewrite
    from metalmind.contracts import TimingSeries
    def series(ratio):
        return TimingSeries(baseline_ms=(ratio,)*30,candidate_ms=(1.,)*30,repetitions=1,order=("baseline_first",)*30)
    iterator = iter([series(1.2),series(.99)])
    monkeypatch.setattr(rewrite,"measure_pair",lambda *a,**kw:next(iterator))
    sleeps = []
    monkeypatch.setattr(rewrite.time,"sleep",sleeps.append)
    outcome, confirmations = rewrite._confirm(lambda:None,lambda:None,ModuleOptimizationConfig())
    assert outcome == "no_win" and len(confirmations) == 2
    assert sleeps == [60]


def test_assumed_domain_is_never_implicitly_selected():
    import pytest
    assert ModuleOptimizationConfig().domain_mode == "checked"
    with pytest.raises(ValueError): ModuleOptimizationConfig(domain_mode="unknown")


def test_cpu_capture_errors_do_not_poison_gpu_health(monkeypatch, tmp_path):
    import metalmind.runtime.worker as worker
    marker = tmp_path / "gpu-fault"
    monkeypatch.setattr(worker,"FAULT_PATH",marker)
    result = optimize_module(Block().eval(),(torch.ones(2,4),))
    assert result.outcome == "unsupported"
    assert not marker.exists()


def test_keyword_calls_use_original_module():
    from metalmind.graph.rewrite import GuardedModule
    module = Block().eval()
    x = torch.ones(2,3)
    captured = capture_module(module,(x,))
    guarded = GuardedModule(module,captured.graph,captured.guard)
    with torch.inference_mode(): torch.testing.assert_close(guarded(x=x),module(x=x))
    assert guarded.last_path == "fallback"


def test_invocation_and_tensor_operator_changes_disable_rewrite(monkeypatch):
    module = Block().eval()
    x = torch.ones(2,3)
    captured = capture_module(module,(x,))
    with torch.inference_mode():
        assert captured.guard.matches((x,))
        monkeypatch.setattr(torch.Tensor,"__add__",lambda a,b: torch.zeros_like(a))
        assert not captured.guard.matches((x,))
    monkeypatch.undo()
    captured = capture_module(module,(x,))
    with torch.inference_mode():
        monkeypatch.setattr(Block,"__call__",lambda self,*args: args[0])
        assert not captured.guard.matches((x,))
