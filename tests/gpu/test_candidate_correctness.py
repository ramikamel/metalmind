import pytest
import torch
from metalmind.contracts import CandidateProposal,CandidateSpec,EvaluationRequest
from metalmind.runtime.worker import evaluate
from metalmind.runtime.abi import signature_for
from metalmind.workloads.registry import workload_for_shape
pytestmark=[pytest.mark.gpu,pytest.mark.skipif(not torch.backends.mps.is_available(),reason="MPS unavailable")]

SOURCE="""
#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x,device const float* bias,
 device float* out,constant uint& B,constant uint& D,constant float& eps,
 uint i [[thread_position_in_grid]]) {
 if(i<B*D) { float z=x[i]+bias[i%D];
 out[i]=0.5f*z*(1.0f+tanh(0.7978845608028654f*(z+0.044715f*z*z*z))); }
}
"""

def candidate_for(source):
    spec=workload_for_shape("bias_gelu",(2,3))
    c=CandidateSpec(candidate_id="test",workload_hash=signature_for(spec),proposal=CandidateProposal(
        source=source,threads=(6,1,1),group_size=(2,1,1),hypothesis="fuse"))
    return spec,c


def test_correct_kernel_passes_all_public_values():
    spec,c=candidate_for(SOURCE)
    result=evaluate(EvaluationRequest(candidate=c,workload=spec))
    assert result.status=="valid",result.reason or result.verification
    assert result.verification.cases==10
    assert not result.elapsed_ms


@pytest.mark.parametrize("source",[SOURCE.replace("out[i]=0.5f*z", "out[i]=0.0f*z"),
    SOURCE.replace("if(i<B*D)","if(i<5)"),
    SOURCE.replace("out[i]=0.5f*z","((device float*)x)[i]=42; out[i]=0.5f*z")])
def test_invalid_kernels_have_no_performance_score(source):
    spec,c=candidate_for(source)
    result=evaluate(EvaluationRequest(candidate=c,workload=spec))
    assert result.status in ("incorrect","compile_error")
    assert not result.elapsed_ms
