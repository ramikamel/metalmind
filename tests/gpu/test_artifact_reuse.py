import pytest
import torch
from metalmind.runtime.execution import build_execution_callable
from metalmind.runtime.metal import MetalBackend
from metalmind.contracts import CandidateSpec
from metalmind.workloads.registry import workload_for_shape,resolve_reference
pytestmark=[pytest.mark.gpu,pytest.mark.skipif(not torch.backends.mps.is_available(),reason="MPS unavailable")]


def test_wrapper_guards_gradients_and_out_of_domain_values():
    spec=workload_for_shape("bias_gelu",(2,3))
    calls=[]
    def fallback(x,b): calls.append(1); return x+b
    # These calls must never compile or dispatch a kernel.
    fn=build_execution_callable(spec,None,fallback=fallback,domain_mode="checked")
    x=torch.ones(2,3,device="mps",requires_grad=True);b=torch.ones(3,device="mps")
    fn(x,b).sum().backward()
    assert x.grad is not None and len(calls)==1
    x=torch.full((2,3),20.,device="mps")
    assert fn(x,b).cpu()[0,0].item()==21
    assert len(calls)==2
