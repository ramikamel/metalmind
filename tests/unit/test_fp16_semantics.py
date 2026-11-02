import pytest
import torch
from metalmind.workloads.registry import get_workload,resolve_reference,resolve_oracle
from metalmind.workloads.fixtures import make_public_fixtures

@pytest.mark.parametrize("family",["bias_gelu","rmsnorm","softmax"])
def test_fp16_uses_float_intermediates_and_final_cast(family):
    spec=get_workload(family,"small","float16")
    fn=resolve_reference(spec.reference_id)
    for fixture in make_public_fixtures(spec):
        output=fn(*fixture.inputs)
        assert output.dtype==torch.float16
        torch.testing.assert_close(output,resolve_oracle(family)(*fixture.inputs).half(),atol=.002,rtol=.02)
