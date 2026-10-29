import pytest
import torch
from metalmind.workloads.registry import get_workload, resolve_reference
from metalmind.workloads.fixtures import make_public_fixtures
pytestmark=[pytest.mark.gpu,pytest.mark.skipif(not torch.backends.mps.is_available(),reason="MPS unavailable")]

@pytest.mark.parametrize("family",["bias_gelu","rmsnorm","softmax"])
def test_cpu_and_mps_references_match(family):
    spec=get_workload(family,"small"); fn=resolve_reference(spec.reference_id)
    fixture=make_public_fixtures(spec)[0]
    actual=fn(*(t.to("mps") for t in fixture.inputs)).cpu()
    torch.testing.assert_close(actual,fn(*fixture.inputs),atol=1e-5,rtol=1e-4)
