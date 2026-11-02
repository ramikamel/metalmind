"""Offline correctness/reuse check. Does not read .env or request a model."""
import json
import sys
from pathlib import Path
import torch
from metalmind.artifacts.loader import load_artifact
from metalmind.workloads.registry import resolve_reference

if __name__=='__main__':
    folder=Path(sys.argv[1]).resolve()
    from metalmind.artifacts.export import verify_artifact_evidence
    manifest=verify_artifact_evidence(folder)
    reference=resolve_reference(manifest.workload.reference_id)
    fn=load_artifact(folder,fallback=reference)
    if fn.optimized is None: raise SystemExit('Optimized artifact was not enabled: '+fn.reason)
    b,d=manifest.workload.inputs[0].shape
    dtype=getattr(torch,manifest.workload.inputs[0].dtype)
    x=torch.randn(b,d,device='mps',dtype=dtype).clamp_(-3,3)
    inputs=(x,) if manifest.workload.family=='softmax' else (x,torch.randn(d,device='mps',dtype=dtype).clamp_(-3,3))
    actual=fn(*inputs);expected=reference(*inputs)
    atol,rtol=(1e-6,1e-5) if manifest.workload.family=='bias_gelu' else (1e-5,1e-4)
    torch.testing.assert_close(actual,expected,atol=atol,rtol=rtol)
    changed=(x[:2].contiguous(),*inputs[1:])
    torch.testing.assert_close(fn(*changed),reference(*changed),atol=atol,rtol=rtol)
    x_grad=x.detach().requires_grad_(True)
    fn(x_grad,*inputs[1:]).sum().backward()
    assert x_grad.grad is not None
    print(json.dumps({'optimized_enabled':True,'new_values_match':True,'changed_shape_fallback':True,'gradient_fallback':True,'max_absolute_error':(actual-expected).abs().max().item()},indent=2))
