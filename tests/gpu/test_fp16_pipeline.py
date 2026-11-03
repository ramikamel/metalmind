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


@pytest.mark.parametrize("family,name", [
    ("bias_gelu", "bias_gelu"),
    ("rmsnorm", "rmsnorm"),
    ("softmax", "row_softmax"),
])
def test_captured_fp16_reference_reaches_gpu_verification(family, name):
    from examples import fp16_workloads
    from metalmind.evaluation import template_proposals
    from metalmind.workloads.capture import capture_function
    from metalmind.workloads.fixtures import make_public_fixtures

    spec = workload_for_shape(family, (8, 17), dtype="float16")
    fn = getattr(fp16_workloads, name)
    captured = capture_function(fn, make_public_fixtures(spec)[0].inputs)
    assert captured.reference_id == f"python:examples.fp16_workloads:{name}"
    candidate = make_candidate(template_proposals(captured)[0].proposal, captured)
    result = evaluate(EvaluationRequest(candidate=candidate, workload=captured))
    assert result.status == "valid", result.reason or result.verification
    assert result.verification.passed


def test_captured_fp16_callable_falls_back_for_fp32_tensors():
    from examples.fp16_workloads import bias_gelu
    from metalmind.evaluation import template_proposals
    from metalmind.runtime.execution import build_execution_callable
    from metalmind.runtime.metal import MetalBackend
    from metalmind.workloads.capture import capture_function

    x = torch.full((8, 17), 0.5, dtype=torch.float16, device="mps")
    bias = torch.full((17,), 0.25, dtype=torch.float16, device="mps")
    spec = capture_function(bias_gelu, (x, bias))
    proposal = template_proposals(spec)[0].proposal
    kernel = MetalBackend().compile(proposal.source, entry_point=proposal.entry_point)
    fallback_inputs = []

    def fallback(*inputs):
        fallback_inputs.append(tuple(t.dtype for t in inputs))
        return bias_gelu(*inputs)

    optimized = build_execution_callable(spec, kernel, fallback=fallback, domain_mode="checked")
    optimized.proposal = proposal
    torch.testing.assert_close(optimized(x, bias), bias_gelu(x, bias), atol=1e-3, rtol=1e-2)
    assert fallback_inputs == []
    torch.testing.assert_close(optimized(x.float(), bias.float()), bias_gelu(x.float(), bias.float()))
    assert fallback_inputs == [(torch.float32, torch.float32)]
