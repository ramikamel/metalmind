"""All-element equivalence plus input, output, and repeated-run integrity."""
import torch
from metalmind.contracts import VerificationSummary
from metalmind.runtime.abi import bind_kernel_args
from metalmind.verification.policy import policy_for
from metalmind.workloads.registry import resolve_reference


def compare_outputs(actual,expected,policy):
    if actual.shape!=expected.shape or actual.dtype!=expected.dtype:
        return VerificationSummary(passed=False,reason="output shape/dtype mismatch")
    a=actual.detach().cpu().double();e=expected.detach().cpu().double()
    if not torch.isfinite(a).all() or not torch.isfinite(e).all():
        return VerificationSummary(passed=False,reason="nonfinite output")
    diff=(a-e).abs();bad=diff>policy.atol+policy.rtol*e.abs()
    return VerificationSummary(passed=not bool(bad.any()),cases=1,
        max_absolute_error=diff.max().item(),max_relative_error=(diff/e.abs().clamp_min(policy.atol)).max().item(),
        rmse=diff.square().mean().sqrt().item(),mismatches=int(bad.sum()),
        reason="numerical mismatch" if bad.any() else None)


def inputs_in_domain(workload,inputs):
    policy=policy_for(workload)
    for i,t in enumerate(inputs):
        limit=policy.max_input if i==0 else policy.max_parameter
        if not bool((torch.isfinite(t)&(t.abs()<=limit)&((t==0)|(t.abs()>=torch.finfo(t.dtype).tiny))).all().item()):
            return False
    return True


def verify_candidate(candidate,workload,fixtures,*,backend,kernel):
    policy=policy_for(workload);reference=resolve_reference(workload.reference_id)
    summaries=[]
    with torch.no_grad():
        for fixture in fixtures:
            cpu_inputs=fixture.inputs
            inputs=tuple(t.to("mps") for t in cpu_inputs)
            originals=tuple(t.clone() for t in inputs)
            count=inputs[0].numel()
            backing=torch.full((count+64,),7316.,dtype=inputs[0].dtype,device="mps")
            out=backing[:count].view(workload.output.shape)
            out.fill_(float("nan"))
            args=bind_kernel_args(workload,inputs,out)
            launch=candidate.proposal
            kernel.function(*args,threads=launch.threads,group_size=launch.group_size)
            backend.synchronize()
            previous=out.cpu().clone()
            summary=compare_outputs(previous,reference(*cpu_inputs),policy)
            if not summary.passed: return summary
            if not torch.equal(backing[count:].cpu(),torch.full((64,),7316.,dtype=out.dtype)):
                return VerificationSummary(passed=False,reason="output guard region modified")
            if any(not torch.equal(a.cpu(),b.cpu()) for a,b in zip(inputs,originals)):
                return VerificationSummary(passed=False,reason="input modified")
            out.fill_(float("nan"))
            kernel.function(*args,threads=launch.threads,group_size=launch.group_size)
            backend.synchronize()
            if not torch.equal(previous,out.cpu()):
                return VerificationSummary(passed=False,reason="nondeterministic output")
            summaries.append(summary)
    return VerificationSummary(passed=True,cases=len(summaries),
        max_absolute_error=max(s.max_absolute_error for s in summaries),
        max_relative_error=max(s.max_relative_error for s in summaries),rmse=max(s.rmse for s in summaries))
