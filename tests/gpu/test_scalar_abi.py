import pytest
import torch
from metalmind.runtime.metal import MetalBackend
from metalmind.runtime.abi import bind_kernel_args, matches_signature
pytestmark=[pytest.mark.gpu,pytest.mark.skipif(not torch.backends.mps.is_available(),reason="MPS unavailable")]


def test_host_owned_abi_scalar_types_and_parameter_values(sample_workload):
    source="""
#include <metal_stdlib>
using namespace metal;
kernel void metalmind_kernel(device const float* x, device const float* bias,
 device float* y, constant uint& B, constant uint& D, constant float& eps,
 uint i [[thread_position_in_grid]]) {
 if(i<B*D) y[i]=x[i]+bias[i%D]+float(B*100+D*10)+eps;
}
"""
    spec=sample_workload.model_copy(update={"epsilon":0.25})
    backend=MetalBackend(); kernel=backend.compile(source,entry_point="metalmind_kernel")
    x=torch.zeros(2,3,device="mps"); bias=torch.tensor([1.,2.,3.],device="mps")
    out=torch.empty_like(x)
    assert matches_signature(spec,(x,bias))
    args=bind_kernel_args(spec,(x,bias),out)
    kernel.function(*args,threads=(6,1,1),group_size=(2,1,1)); backend.synchronize()
    torch.testing.assert_close(out.cpu(),torch.tensor([[231.25,232.25,233.25]]).expand(2,3))
    bias.add_(2)
    assert matches_signature(spec,(x,bias))
    with torch.no_grad():
        kernel.function(*bind_kernel_args(spec,(x,bias),out),threads=(6,1,1),group_size=(2,1,1))
    backend.synchronize()
    assert out.cpu()[0,0].item()==233.25
