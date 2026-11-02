import pytest
import torch
from metalmind.workloads.registry import workload_for_shape
from metalmind.contracts import CandidateProposal,EvaluationRequest
from metalmind.agent.candidates import make_candidate
from metalmind.runtime.worker import evaluate
pytestmark=[pytest.mark.gpu,pytest.mark.skipif(not torch.backends.mps.is_available(),reason="MPS unavailable")]


def test_fp16_kernel_has_fp32_intermediates_and_passes_all_values():
    spec=workload_for_shape("bias_gelu",(2,3),dtype="float16")
    source="""
#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const half* x,device const half* b,device half* y,
 constant uint& B,constant uint& D,constant float& eps,uint i [[thread_position_in_grid]]) {
 if(i<B*D) {float z=float(x[i])+float(b[i%D]);y[i]=half(0.5f*z*(1.0f+tanh(0.7978845608f*(z+0.044715f*z*z*z))));}
}
"""
    c=make_candidate(CandidateProposal(source=source,threads=(6,1,1),group_size=(2,1,1),hypothesis="fuse"),spec)
    result=evaluate(EvaluationRequest(candidate=c,workload=spec))
    assert result.status=="valid",result.reason or result.verification


@pytest.mark.parametrize("family",["bias_gelu","rmsnorm","softmax"])
def test_all_fp16_template_families_verify(family):
    from metalmind.evaluation import template_proposals
    spec=workload_for_shape(family,(2,3),dtype="float16")
    c=make_candidate(template_proposals(spec)[0].proposal,spec)
    result=evaluate(EvaluationRequest(candidate=c,workload=spec))
    assert result.status=="valid",result.reason or result.verification
