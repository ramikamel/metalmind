"""Shared full-output and integrity checks for module candidates and baselines."""
import torch

def _compare(actual, expected):
    if isinstance(expected,torch.Tensor):
        if not isinstance(actual,torch.Tensor) or not torch.isfinite(actual).all().item() or not torch.isfinite(expected).all().item():
            raise ValueError("Whole-module outputs must be finite matching tensors")
        torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-4,equal_nan=False)
        return (actual-expected).abs().max().item() if actual.numel() else 0.
    if type(actual) is not type(expected): raise ValueError("Output structure changed")
    if isinstance(expected,dict):
        if list(actual) != list(expected): raise ValueError("Output keys changed")
        return max((_compare(actual[k],v) for k,v in expected.items()),default=0.)
    if isinstance(expected,(tuple,list)):
        if len(actual) != len(expected): raise ValueError("Output length changed")
        return max((_compare(a,b) for a,b in zip(actual,expected)),default=0.)
    if actual != expected: raise ValueError("Non-tensor output changed")
    return 0.


def _versions(tensors):
    values = []
    for t in tensors:
        try: values.append(t._version)
        except RuntimeError: values.append(None)
    return tuple(values)


def make_private_inputs(examples, seed):
    rng=torch.Generator(device="cpu").manual_seed(seed)
    return tuple((torch.randn(tuple(t.shape),generator=rng)*.25).clamp_(-1,1).to(t.device) for t in examples)
