"""Optional fresh correctness check; never a new performance measurement."""
from pathlib import Path


def check_artifact_reuse(path):
    import torch
    if not torch.backends.mps.is_available(): return {"outcome":"unavailable","reason":"Fresh GPU check requires MPS"}
    from metalmind.artifacts.export import verify_artifact_evidence
    from metalmind.artifacts.loader import load_artifact
    from metalmind.workloads.registry import resolve_reference
    from metalmind.verification.policy import policy_for
    from metalmind.runtime.lock import gpu_lock
    from metalmind.runtime.worker import gpu_faulted, _record_fault
    if gpu_faulted(): return {"outcome":"unavailable","reason":"GPU health probe required before execution"}
    with gpu_lock():
        manifest=verify_artifact_evidence(path)
        if manifest.workload.reference_id!=manifest.workload.family:
            return {"outcome":"unavailable","reason":"Demo checks support registered semantic references only"}
        reference=resolve_reference(manifest.workload.reference_id)
        fn=load_artifact(path,fallback=reference)
        if fn.optimized is None: return {"outcome":"unavailable","reason":fn.reason}
        policy=policy_for(manifest.workload)
        b,d=manifest.workload.inputs[0].shape;dtype=getattr(torch,manifest.workload.inputs[0].dtype)
        x=torch.linspace(-.5,.5,b*d,dtype=dtype).reshape(b,d).to("mps")
        inputs=(x,) if manifest.workload.family=="softmax" else (x,torch.linspace(-.05,.05,d,dtype=dtype).to("mps"))
        try:
            with torch.no_grad():
                actual=fn(*inputs);expected=reference(*inputs)
                torch.testing.assert_close(actual,expected,atol=policy.atol,rtol=policy.rtol)
                changed=(x[:max(0,b-1)].contiguous(),*inputs[1:])
                torch.testing.assert_close(fn(*changed),reference(*changed),atol=policy.atol,rtol=policy.rtol)
                error=(actual-expected).abs().max().item()
            grad_input=x.detach().requires_grad_(True)
            fn(grad_input,*inputs[1:]).sum().backward()
            if grad_input.grad is None: raise AssertionError("Gradient fallback did not preserve autograd")
            return {"outcome":"passed","artifact_id":manifest.artifact_id,"optimized_enabled":True,"new_values_match":True,
                "changed_shape_fallback":True,"gradient_fallback":True,"max_absolute_error":error,"new_performance_measurement":False}
        except AssertionError as exc: return {"outcome":"failed","reason":str(exc)[:500]}
        except RuntimeError as exc:
            _record_fault(str(exc));return {"outcome":"failed","reason":"GPU/runtime failure; health probe required"}
