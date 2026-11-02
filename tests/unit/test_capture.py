import torch
from examples.my_workloads import bias_gelu,exact_gelu,row_softmax,rmsnorm
from metalmind.workloads.capture import capture_function


def test_supported_importable_functions_are_captured():
    x=torch.zeros(2,3);parameter=torch.ones(3)
    for fn,inputs,family in [(bias_gelu,(x,parameter),'bias_gelu'),(rmsnorm,(x,parameter),'rmsnorm'),(row_softmax,(x,),'softmax')]:
        result=capture_function(fn,inputs)
        assert result.family==family
        assert result.reference_id.startswith('python:examples.my_workloads:')


def test_exact_gelu_and_local_closures_are_rejected():
    x=torch.zeros(2,3);b=torch.ones(3)
    assert capture_function(exact_gelu,(x,b)).reason
    assert capture_function(lambda x:x,(x,)).reason


def changes_rng(x):
    torch.manual_seed(3)
    return torch.softmax(x,dim=-1)


def test_capture_rejects_python_side_effects_before_tracing():
    before=torch.random.get_rng_state().clone()
    result=capture_function(changes_rng,(torch.zeros(2,3),))
    assert result.reason
    assert torch.equal(torch.random.get_rng_state(),before)
