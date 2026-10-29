import pytest
import torch
from metalmind.contracts import EvaluationRequest, CandidateSpec, CandidateProposal
from metalmind.runtime.worker import evaluate
from metalmind.runtime.abi import signature_for
from metalmind.workloads.registry import workload_for_shape
pytestmark=[pytest.mark.gpu,pytest.mark.skipif(not torch.backends.mps.is_available(),reason="MPS unavailable")]


def test_spawned_worker_reports_compile_error():
    spec=workload_for_shape("bias_gelu",(2,3))
    c=CandidateSpec(candidate_id="bad",workload_hash=signature_for(spec),proposal=CandidateProposal(
        source="not valid Metal",threads=(6,1,1),group_size=(2,1,1),hypothesis="test"))
    result=evaluate(EvaluationRequest(candidate=c,workload=spec))
    assert result.status=="compile_error"
