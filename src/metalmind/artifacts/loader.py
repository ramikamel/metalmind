"""Offline artifacts retain the original callable for incompatible inputs."""
from pathlib import Path
from metalmind.artifacts.export import verify_artifact_evidence,environment_fingerprint
from metalmind.runtime.execution import build_execution_callable
from metalmind.runtime.metal import MetalBackend
from metalmind.workloads.registry import resolve_reference,fingerprint_reference

class OptimizedFunction:
    def __init__(self,fallback,*,optimized=None,reason="fallback only"):
        self.fallback=fallback;self.optimized=optimized;self.reason=reason
    def __call__(self,*inputs):
        return self.optimized(*inputs) if self.optimized else self.fallback(*inputs)

def load_artifact(path,*,fallback):
    try:
        manifest=verify_artifact_evidence(path)
        if manifest.environment_fingerprint!=environment_fingerprint(): raise ValueError("environment changed; revalidation required")
        from metalmind.workloads.capture import describe_reference
        expected=manifest.workload.reference_id
        if expected.startswith("python:"):
            if describe_reference(fallback)!=expected: raise ValueError("fallback reference identity mismatch")
        elif fallback is not resolve_reference(expected): raise ValueError("fallback reference identity mismatch")
        if fingerprint_reference(manifest.workload.reference_id)!=manifest.workload.reference_hash: raise ValueError("reference source changed")
        backend=MetalBackend();kernel=backend.compile(manifest.candidate.proposal.source,entry_point="metalmind_kernel")
        fn=build_execution_callable(manifest.workload,kernel,fallback=fallback,domain_mode=manifest.domain_mode)
        fn.proposal=manifest.candidate.proposal
        return OptimizedFunction(fallback,optimized=fn,reason="matching verified artifact")
    except (ValueError,OSError,RuntimeError,KeyError,TypeError) as exc:
        return OptimizedFunction(fallback,reason=str(exc))

def revalidate_artifact(path,*,fallback,config):
    from metalmind.agent.controller import RunSummary
    from metalmind.artifacts.store import ArtifactStore
    from metalmind.artifacts.export import finalize_run
    from metalmind.runtime.environment import probe_environment
    manifest=verify_artifact_evidence(path)
    if config.domain_mode!=manifest.domain_mode: raise ValueError("Revalidation requires the same explicitly selected domain mode")
    if fallback is not resolve_reference(manifest.workload.reference_id): raise ValueError("fallback mismatch")
    store=ArtifactStore.create_run(config.runs_dir,config,probe_environment())
    store.write_source(manifest.candidate.candidate_id,manifest.candidate.proposal.source)
    store.write_manifest({**store.read_manifest(),"elapsed_seconds":0,"parent_artifact":str(path),
        "workload":manifest.workload.model_dump(mode="json"),"requests":0,"output_tokens":0,
        "archive":[{"score":1.05,"candidate":manifest.candidate.model_dump(mode="json")}],
        "evidence_kind":"revalidation; no model generation or search"})
    return finalize_run(RunSummary(workload=manifest.workload,config=config,run_path=store.path,
        outcome="revalidate",incumbent=manifest.candidate,incumbent_score=1.05))
