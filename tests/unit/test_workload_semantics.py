import math
import pytest
import torch
from metalmind.workloads.registry import get_workload, resolve_reference, resolve_oracle
from metalmind.workloads.fixtures import make_public_fixtures, make_private_fixtures


def test_rms_is_not_centered_and_epsilon_is_inside_sqrt():
    spec=get_workload("rmsnorm","small")
    fn=resolve_reference(spec.reference_id)
    x=torch.tensor([[3.,4.]]); w=torch.tensor([2.,3.])
    torch.testing.assert_close(fn(x,w),torch.tensor([[6/math.sqrt(12.50001),12/math.sqrt(12.50001)]]))


def test_softmax_uses_final_axis_and_stable_shift():
    fn=resolve_reference(get_workload("softmax","small").reference_id)
    x=torch.tensor([[0.,math.log(3.)],[80.,80.]])
    torch.testing.assert_close(fn(x),torch.tensor([[.25,.75],[.5,.5]]))


def test_bias_is_runtime_parameter_and_tanh_gelu():
    fn=resolve_reference(get_workload("bias_gelu","small").reference_id)
    x=torch.zeros(2,3)
    a=fn(x,torch.zeros(3)); b=fn(x,torch.ones(3))
    assert torch.equal(a,torch.zeros(2,3))
    torch.testing.assert_close(b,torch.full((2,3),0.8411919906))


@pytest.mark.parametrize("family",["bias_gelu","rmsnorm","softmax"])
def test_references_agree_with_independent_oracle_and_fixture_domains(family):
    spec=get_workload(family,"small")
    fn=resolve_reference(spec.reference_id); oracle=resolve_oracle(family)
    fixtures=make_public_fixtures(spec)
    assert len(fixtures)==10
    for f in fixtures:
        expected=oracle(*f.inputs).float()
        torch.testing.assert_close(fn(*f.inputs),expected,atol=1e-5,rtol=1e-4)
        assert all(torch.isfinite(t).all() for t in f.inputs)
    private=make_private_fixtures(spec,tuple(range(100,116)))
    assert len(private)==22
    assert not hasattr(spec,"seeds")
