"""Promotion is conditional on frozen source, private verification and two sessions."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Literal
import math
import os
import shutil
import tempfile
import secrets
import time
from metalmind.api import OptimizationResult
from metalmind.artifacts.store import atomic_json,ArtifactStore
from metalmind.benchmarking.promotion import decide_promotion
from metalmind.contracts import Contract,WorkloadSpec,CandidateSpec,VerificationSummary,TimingReport,EvaluationRequest
from metalmind.runtime.worker import evaluate
from metalmind.runtime.environment import probe_environment
from metalmind.verification.final_gate import run_final_gate

class ArtifactManifest(Contract):
    artifact_id: str
    workload: WorkloadSpec
    candidate: CandidateSpec
    environment_fingerprint: str
    source_hash: str
    evidence_hash: str
    domain_mode: Literal["checked","assume"]
    abi_version: Literal[1] = 1

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def _hardware_profile():
    import subprocess
    import torch
    try:
        raw=json.loads(subprocess.check_output(['system_profiler','SPDisplaysDataType','-json'],text=True,stderr=subprocess.DEVNULL,timeout=20))
        gpus=[g for g in raw.get('SPDisplaysDataType',[]) if g.get('spdisplays_vendor')=='sppci_vendor_Apple']
        if len(gpus)!=1 or not gpus[0].get('sppci_cores'): raise ValueError('GPU identity unavailable')
        gpu=gpus[0]
        return {'model':gpu.get('sppci_model'),'cores':gpu['sppci_cores'],
            'metal_family':gpu.get('spdisplays_mtlgpufamilysupport'),
            'recommended_working_set':torch.mps.recommended_max_memory()}
    except (OSError,subprocess.SubprocessError,ValueError,RuntimeError) as exc:
        raise ValueError('Cannot establish GPU configuration; artifact reuse remains disabled') from None


def environment_fingerprint():
    import os
    profile=_hardware_profile()
    flags={name:os.environ.get(name) for name in ('PYTORCH_MPS_FAST_MATH','PYTORCH_MPS_PREFER_METAL','PYTORCH_MPS_METAL_VERSION','PYTORCH_ENABLE_MPS_FALLBACK','MTL_SHADER_VALIDATION')}
    return digest({'environment':asdict(probe_environment()),'gpu':profile,'flags':flags})


def _validate_confirmation_evidence(verification,confirmations,*,domain_mode,required_baselines):
    from metalmind.benchmarking.statistics import summarize_pairs
    if not verification.passed or verification.cases!=22 or verification.mismatches:
        raise ValueError("a complete private verification gate is required")
    if len(confirmations)!=2: raise ValueError("two confirmation sessions required")
    sets=[];sessions=[]
    for report in confirmations:
        names=[c.name for c in report.comparisons]
        if not names or len(names)!=len(set(names)) or report.domain_mode!=domain_mode:
            raise ValueError("baseline identity/domain mismatch")
        sets.append(set(names));session=[]
        for comparison in report.comparisons:
            series=comparison.series
            if len(series.baseline_ms)!=len(series.candidate_ms) or len(series.baseline_ms)<30 or series.repetitions<1 or len(series.order)!=len(series.baseline_ms):
                raise ValueError("incomplete raw timing evidence")
            observed=summarize_pairs(series.baseline_ms,series.candidate_ms,seed=report.seed,resamples=report.bootstrap_resamples)
            claimed=comparison.summary
            if claimed.pairs!=observed.pairs or any(not math.isclose(getattr(observed,k),getattr(claimed,k),rel_tol=1e-10,abs_tol=1e-12) for k in ('speedup','lower','upper')):
                raise ValueError("summary is inconsistent with raw timing samples")
            session.append(observed)
        sessions.append(session)
    if sets[0]!=sets[1] or not set(required_baselines).issubset(sets[0]) or 'original_eager' not in sets[0]:
        raise ValueError("required baseline set differs across sessions")
    if not decide_promotion(verification,sessions).promote: raise ValueError("promotion evidence does not pass")


def _artifact_identity(candidate,workload,fingerprint,domain_mode,evidence):
    return digest({"candidate":candidate.model_dump(mode="json"),"workload":workload.model_dump(mode="json"),
        "environment":fingerprint,"domain_mode":domain_mode,"abi_version":1,"evidence":evidence})


def export_artifact(candidate,workload,verification,confirmations,*,root,domain_mode,required_baselines=None,deadline=None):
    from metalmind.runtime.abi import signature_for
    from metalmind.agent.candidates import make_candidate,validate_proposal
    validate_proposal(candidate.proposal.model_dump_json())
    if deadline is not None and time.monotonic()>=deadline: raise TimeoutError("export deadline expired")
    required=tuple(sorted(required_baselines or [c.name for c in confirmations[0].comparisons]))
    _validate_confirmation_evidence(verification,confirmations,domain_mode=domain_mode,required_baselines=required)
    if candidate.workload_hash!=signature_for(workload) or make_candidate(candidate.proposal,workload).candidate_id!=candidate.candidate_id:
        raise ValueError("candidate identity/workload mismatch")
    evidence={"verification":verification.model_dump(mode="json"),"confirmations":[r.model_dump(mode="json") for r in confirmations],"required_baselines":required}
    fingerprint=environment_fingerprint()
    identity=_artifact_identity(candidate,workload,fingerprint,domain_mode,evidence)
    root=Path(root);root.mkdir(parents=True,exist_ok=True);folder=root/identity
    staging=Path(tempfile.mkdtemp(dir=root,prefix='.pending-'))
    try:
        source_hash=hashlib.sha256(candidate.proposal.source.encode()).hexdigest()
        manifest=ArtifactManifest(artifact_id=identity,workload=workload,candidate=candidate,
            environment_fingerprint=fingerprint,source_hash=source_hash,evidence_hash=digest(evidence),domain_mode=domain_mode)
        (staging/"kernel.metal").write_text(candidate.proposal.source)
        atomic_json(staging/"evidence.json",evidence);atomic_json(staging/"manifest.json",manifest.model_dump(mode="json"))
        if deadline is not None and time.monotonic()>=deadline: raise TimeoutError("export deadline expired")
        os.rename(staging,folder)
        if deadline is not None and time.monotonic()>=deadline:
            shutil.rmtree(folder);raise TimeoutError("export exceeded job deadline")
        return folder
    finally:
        if staging.exists(): shutil.rmtree(staging)


def verify_artifact_evidence(path):
    from metalmind.runtime.abi import signature_for
    from metalmind.agent.candidates import make_candidate
    folder=Path(path);manifest=ArtifactManifest.model_validate_json((folder/"manifest.json").read_text())
    from metalmind.agent.candidates import validate_proposal
    validate_proposal(manifest.candidate.proposal.model_dump_json())
    source=(folder/"kernel.metal").read_text();evidence=json.loads((folder/"evidence.json").read_text())
    if source!=manifest.candidate.proposal.source or hashlib.sha256(source.encode()).hexdigest()!=manifest.source_hash or digest(evidence)!=manifest.evidence_hash:
        raise ValueError("artifact hash mismatch")
    if manifest.candidate.workload_hash!=signature_for(manifest.workload) or make_candidate(manifest.candidate.proposal,manifest.workload).candidate_id!=manifest.candidate.candidate_id:
        raise ValueError("candidate identity/guards mismatch")
    identity=_artifact_identity(manifest.candidate,manifest.workload,manifest.environment_fingerprint,manifest.domain_mode,evidence)
    if identity!=manifest.artifact_id: raise ValueError("artifact identity mismatch")
    verification=VerificationSummary.model_validate(evidence['verification'])
    reports=[TimingReport.model_validate(r) for r in evidence['confirmations']]
    _validate_confirmation_evidence(verification,reports,domain_mode=manifest.domain_mode,required_baselines=evidence['required_baselines'])
    return manifest


def finalize_run(summary):
    candidate=summary.incumbent;config=summary.config;store=ArtifactStore(summary.run_path)
    existing=summary.run_path/"result.json"
    if existing.exists():
        prior=json.loads(existing.read_text())
        return OptimizationResult(prior['outcome'],summary.run_path,Path(prior['artifact_path']) if prior['artifact_path'] else None,prior['reason'])
    def finish(outcome,reason,artifact_path=None):
        result=OptimizationResult(outcome,summary.run_path,artifact_path,reason)
        atomic_json(summary.run_path/"result.json",{"outcome":outcome,"reason":reason,"artifact_path":str(artifact_path) if artifact_path else None})
        store.append_event({"event":"finalized","outcome":outcome,"reason":reason})
        manifest=store.read_manifest() if (store.path/"manifest.json").exists() else {}
        manifest.update({"phase":"terminal","outcome":outcome})
        store.write_manifest(manifest)
        return result
    if candidate is None: return finish(summary.outcome,"No verified measured candidate was selected")
    if summary.outcome in ("cancelled","failed"): return finish(summary.outcome,"Search stopped before finalization")
    if summary.incumbent_score<1.05: return finish("no_win","Search did not establish the minimum 1.05x public-wrapper gain")
    previous=store.read_manifest()
    allowance=min(config.wall_seconds-previous['elapsed_seconds'],previous.get('wall_deadline_utc',time.time()+config.wall_seconds)-time.time())
    deadline=time.monotonic()+max(0,allowance)
    previous['phase']='finalizing';store.write_manifest(previous)
    required_baselines={'original_eager'}
    for event in store.read_events():
        if event.get('event')=='evaluation' and event.get('candidate_id')==candidate.candidate_id and event['result'].get('timing'):
            required_baselines.update(c['name'] for c in event['result']['timing']['comparisons'])
    def remaining(): return deadline-time.monotonic()
    if remaining()<=0: return finish("budget_exhausted","No finalization time remaining")
    seeds=tuple(secrets.randbits(63) for _ in range(16))
    gate=run_final_gate(candidate,summary.workload,seeds,marker_path=summary.run_path/"private-gate-consumed",
        timeout=min(config.execution_timeout_seconds,remaining()))
    atomic_json(summary.run_path/"private-verification.json",gate.model_dump(mode="json"))
    if gate.status!="valid": return finish("failed","Private verification did not pass; gate is closed")
    confirmations=[]
    for session in range(2):
        if session:
            if remaining()<=config.confirmation_cooldown_seconds: return finish("budget_exhausted","Insufficient time for independent confirmation")
            time.sleep(config.confirmation_cooldown_seconds)
        if remaining()<=0: return finish("budget_exhausted","Confirmation deadline exhausted")
        result=evaluate(EvaluationRequest(candidate=candidate,workload=summary.workload,mode="confirm",measure=True,
            domain_mode=config.domain_mode,seed=config.seed+session+1,total_timeout_seconds=max(.001,remaining()),
            warmups=config.warmups,confirmation_pairs=config.confirmation_pairs,bootstrap_resamples=config.bootstrap_resamples,
            compile_timeout_seconds=min(config.compile_timeout_seconds,remaining()),
            execution_timeout_seconds=min(config.execution_timeout_seconds,remaining())))
        atomic_json(summary.run_path/f"confirmation-{session+1}.json",result.model_dump(mode="json"))
        if remaining()<=0: return finish("budget_exhausted","Confirmation arrived after job deadline")
        if result.status!="valid" or result.timing is None: return finish("inconclusive","Confirmation did not yield stable valid measurements")
        confirmations.append(result.timing)
        if any(c.summary.speedup<1.05 or c.summary.lower<=1.0 for c in result.timing.comparisons):
            return finish("no_win","Independent confirmation did not establish a reliable gain")
    if remaining()<=0: return finish("budget_exhausted","No time remains for promotion")
    try:
        folder=export_artifact(candidate,summary.workload,gate.verification,confirmations,root=config.artifacts_dir,
            domain_mode=config.domain_mode,required_baselines=required_baselines,deadline=deadline)
    except TimeoutError: return finish("budget_exhausted","Promotion export exceeded job deadline")
    except ValueError as exc: return finish("inconclusive",str(exc))
    return finish("promoted","Numerically verified and faster in both independent confirmation sessions",folder)
