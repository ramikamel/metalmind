"""Each available framework baseline is verified before timing."""
import os
import torch
import torch.nn.functional as F
from metalmind.workloads.registry import resolve_reference
from metalmind.workloads.fixtures import make_public_fixtures
from metalmind.verification.numerics import compare_outputs
from metalmind.verification.policy import policy_for


def collect_baselines(workload,environment=None):
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")=="1":
        raise ValueError("Disable PYTORCH_ENABLE_MPS_FALLBACK for GPU comparisons")
    reference=resolve_reference(workload.reference_id)
    options=[("original_eager",reference)]
    if workload.family=="rmsnorm":
        def idiomatic(x,weight):
            return F.rms_norm(x.float(),(x.shape[-1],),weight.float(),eps=1e-5).to(x.dtype)
        options.append(("functional_rms_norm",idiomatic))
    try:
        compiled=torch.compile(reference,fullgraph=True)
        options.append(("torch_compile_inductor_fullgraph",compiled))
    except Exception as exc:
        compile_error=f"torch.compile construction: {type(exc).__name__}"
    else: compile_error=None
    valid=[];unavailable=[] if compile_error is None else [compile_error]
    fixtures=make_public_fixtures(workload)
    for name,fn in options:
        try:
            with torch.no_grad():
                for fixture in fixtures:
                    inputs=tuple(t.to("mps") for t in fixture.inputs)
                    actual=fn(*inputs);torch.mps.synchronize()
                    if not compare_outputs(actual,reference(*fixture.inputs),policy_for(workload)).passed:
                        raise ValueError("baseline disagrees with numerical contract")
            valid.append((name,fn))
        except Exception as exc:
            unavailable.append(f"{name}: {type(exc).__name__}: {str(exc)[:300]}")
    if not any(n=="original_eager" for n,_ in valid):
        raise ValueError("original reference baseline did not validate")
    return valid,tuple(unavailable)
