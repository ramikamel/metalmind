import torch
from metalmind.verification.numerics import compare_outputs
from metalmind.verification.policy import CorrectnessPolicy


def test_tolerance_near_zero_and_nonfinite_rejection():
    policy=CorrectnessPolicy(atol=1e-6,rtol=1e-5,max_input=10,max_parameter=10)
    assert compare_outputs(torch.tensor([1e-7]),torch.tensor([0.]),policy).passed
    assert not compare_outputs(torch.tensor([1e-3]),torch.tensor([0.]),policy).passed
    assert not compare_outputs(torch.tensor([float('nan')]),torch.tensor([0.]),policy).passed
