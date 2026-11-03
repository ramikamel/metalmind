"""Frozen sequential campaigns with explicit denominators and strategy labels."""
import fcntl
import itertools
import json
import math
import random
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from pydantic import Field
from metalmind.contracts import Contract,CandidateProposal
from metalmind.config import RunConfig
from metalmind.demonstration.provenance import runtime_source_info
from metalmind.artifacts.export import digest,environment_fingerprint,finalize_run
from metalmind.artifacts.store import atomic_json
from metalmind.agent.controller import SearchController
from metalmind.agent.meta import MetaProvider
from metalmind.agent.provider import ReplayProvider,ProposalResponse,UsageRecord
from metalmind.workloads.registry import get_workload
from metalmind.verification.policy import policy_for

class CampaignConfig(Contract):
    families: tuple[Literal['bias_gelu','rmsnorm','softmax'],...] = ('bias_gelu','rmsnorm','softmax')
    cases: tuple[Literal['small','medium','large'],...] = ('small','medium','large')
    strategies: tuple[Literal['feedback','one_shot','independent','template'],...] = ('feedback','one_shot','independent','template')
    trials: int = Field(default=3,ge=1)
    dtype: Literal['float32','float16'] = 'float32'
    domain_mode: Literal['checked','assume'] = 'checked'
    run_config: RunConfig = RunConfig()

def freeze_campaign(config):
    spec=config.model_dump(mode='json')
    workloads=[]
    for family,case in itertools.product(config.families,config.cases):
        workload=get_workload(family,case,config.dtype)
        workloads.append({'family':family,'case':case,'workload':workload.model_dump(mode='json'),
            'policy':policy_for(workload).model_dump(mode='json')})
    frozen={'manifest_version':2,'config':spec,'workloads':workloads}
    return {'campaign_id':digest(frozen),**frozen,'jobs':len(config.families)*len(config.cases)*len(config.strategies)*config.trials}

def summarize_campaign(rows):
    return {'attempted':len(rows),'promoted':sum(r['outcome']=='promoted' for r in rows),
        'failed':sum(r['outcome']=='failed' for r in rows),
        'geometric_mean_speedup':math.exp(sum(math.log(r.get('speedup',1.0)) for r in rows)/len(rows)) if rows else None,
        'rows':rows}

class IndependentProvider:
    def __init__(self,provider): self.provider=provider
    def propose(self,context): return self.provider.propose(context.model_copy(update={'feedback':(),'incumbent_source':None}))

def template_proposals(workload):
    dtype='float' if workload.inputs[0].dtype=='float32' else 'half'
    parameter='' if workload.family=='softmax' else f'device const {dtype}* p,'
    prefix=f'#include <metal_stdlib>\nusing namespace metal;\nkernel void metalmind_kernel(device const {dtype}* x,{parameter}device {dtype}* y,constant uint& B,constant uint& D,constant float& eps,uint i [[thread_position_in_grid]]) {{'
    if workload.family=='bias_gelu':
        body='if(i>=B*D)return;float z=float(x[i])+float(p[i%D]);y[i]='+dtype+'(0.5f*z*(1+tanh(0.7978845608f*(z+0.044715f*z*z*z))));}'
        threads=(workload.inputs[0].shape[0]*workload.inputs[0].shape[1],1,1)
    elif workload.family=='rmsnorm':
        body='if(i>=B)return;float s=0;for(uint j=0;j<D;j++){float v=float(x[i*D+j]);s+=v*v;}float scale=rsqrt(s/float(D)+eps);for(uint j=0;j<D;j++)y[i*D+j]='+dtype+'(float(x[i*D+j])*scale*float(p[j]));}'
        threads=(workload.inputs[0].shape[0],1,1)
    else:
        body='if(i>=B)return;float m=-INFINITY;for(uint j=0;j<D;j++)m=max(m,float(x[i*D+j]));float s=0;for(uint j=0;j<D;j++)s+=exp(float(x[i*D+j])-m);for(uint j=0;j<D;j++)y[i*D+j]='+dtype+'(exp(float(x[i*D+j])-m)/s);}'
        threads=(workload.inputs[0].shape[0],1,1)
    return [ProposalResponse(proposal=CandidateProposal(source=prefix+body,threads=threads,group_size=(g,1,1),hypothesis='Predeclared template, tune group size'),model='template',usage=UsageRecord(input_tokens=0,output_tokens=0)) for g in (32,64,128,256)]

def _job_order(config):
    jobs = [
        {"family": family, "case": case, "strategy": strategy, "trial": trial}
        for family, case, strategy, trial in itertools.product(
            config.families, config.cases, config.strategies, range(config.trials)
        )
    ]
    random.Random(config.run_config.seed).shuffle(jobs)
    return jobs


def _job_budget(strategy, config):
    if strategy == "template":
        return 0
    return 1 if strategy == "one_shot" else config.run_config.max_requests


@contextmanager
def _campaign_lock(folder):
    with (folder / "campaign.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Campaign already has an active writer") from None
        yield


def _read_json(path):
    return json.loads(path.read_text())


def _run_checkpoint(job_root):
    manifests = list(job_root.glob("*/manifest.json"))
    if len(manifests) > 1:
        raise ValueError("Campaign job has multiple search directories")
    if not manifests:
        return None, {}
    return manifests[0].parent, _read_json(manifests[0])


def _failed_row(job, job_root, reason, accounting):
    run_path, manifest = _run_checkpoint(job_root)
    return {
        **job,
        "outcome": "failed",
        "speedup": 1.0,
        "reason": reason,
        "run_path": str(run_path) if run_path else None,
        "proposal_slots": manifest.get("requests", 0),
        "request_accounting": accounting,
    }


def _execute_job(job, config, job_root):
    strategy = job["strategy"]
    workload = get_workload(job["family"], job["case"], config.dtype)
    cfg = config.run_config.model_copy(update={
        "domain_mode": config.domain_mode,
        "seed": config.run_config.seed + job["trial"],
        "runs_dir": job_root,
    })
    if strategy == "one_shot":
        cfg = cfg.model_copy(update={"max_candidates": 1, "max_requests": 1})
    try:
        if strategy == "template":
            provider = ReplayProvider(template_proposals(workload))
        elif strategy == "independent":
            provider = IndependentProvider(MetaProvider(cfg))
        else:
            provider = None
        summary = SearchController(provider=provider).run(workload, cfg)
        result = finalize_run(summary)
        return _finished_row(job, summary.run_path, summary.requests, result)
    except Exception as exc:
        return _failed_row(job, job_root, str(exc)[:500], "exception_upper_bound")


def _finished_row(job, run_path, slots, result):
    speedup = 1.0
    if result.artifact_path:
        evidence = _read_json(Path(result.artifact_path) / "evidence.json")
        speedup = min(
            comparison["summary"]["speedup"]
            for session in evidence["confirmations"]
            for comparison in session["comparisons"]
        )
    return {
        **job,
        "outcome": result.outcome,
        "speedup": speedup,
        "reason": result.reason,
        "run_path": str(run_path),
        "artifact_path": str(result.artifact_path) if result.artifact_path else None,
        "proposal_slots": slots,
        "request_accounting": "search_ledger",
    }


def _recover_job(job, job_root):
    run_path, manifest = _run_checkpoint(job_root)
    if run_path and (run_path / "result.json").is_file():
        result = SimpleNamespace(**_read_json(run_path / "result.json"))
        return _finished_row(job, run_path, manifest["requests"], result)
    return _failed_row(
        job, job_root, "Interrupted before the campaign checkpoint", "interrupted_upper_bound"
    )


def _with_accounting(row):
    # The search controller also uses request slots to bound local template tuning.
    slots = row["proposal_slots"]
    return {
        **row,
        "requests": slots,
        "model_requests": 0 if row["strategy"] == "template" else slots,
    }


def run_campaign(config, *, output_root=Path("runs/campaigns"), resume=False, max_jobs=None):
    from metalmind.runtime.worker import gpu_faulted

    if max_jobs is not None and max_jobs < 1:
        raise ValueError("max_jobs must be positive")
    frozen = freeze_campaign(config)
    identity = frozen["campaign_id"]
    folder = Path(output_root) / identity
    if resume and not folder.is_dir():
        raise ValueError("Campaign does not exist for this configuration")
    jobs = _job_order(config)
    execution = {
        "execution_version": 1,
        "source_sha256": runtime_source_info()["sha256"],
        "environment_fingerprint": environment_fingerprint(),
        "job_order": jobs,
        "max_model_requests": sum(_job_budget(job["strategy"], config) for job in jobs),
    }
    manifest = {**frozen, "execution": execution}
    manifest_path = folder / "campaign.json"
    summary_path = folder / "summary.json"
    active_path = folder / "active.json"

    folder.mkdir(parents=True, exist_ok=True)
    with _campaign_lock(folder):
        if manifest_path.exists():
            if not resume:
                raise ValueError("Campaign already exists; use resume to continue")
            if _read_json(manifest_path) != manifest:
                raise ValueError("Frozen campaign, package source, or environment differs from the original execution")
        else:
            if summary_path.exists() or active_path.exists():
                raise ValueError("Campaign execution state is missing its frozen manifest")
            atomic_json(manifest_path, manifest)

        rows = _read_json(summary_path)["rows"] if summary_path.exists() else []
        if len(rows) > len(jobs) or any(
            {key: row[key] for key in job} != job for row, job in zip(rows, jobs)
        ):
            raise ValueError("Campaign results do not match the frozen job prefix")

        def checkpoint():
            report = {
                **summarize_campaign(rows),
                "campaign_id": identity,
                "planned_jobs": len(jobs),
                "complete": len(rows) == len(jobs),
                "max_model_requests": execution["max_model_requests"],
                "model_requests": sum(row["model_requests"] for row in rows),
            }
            atomic_json(summary_path, report)
            return report

        def job_root(index):
            return config.run_config.runs_dir / identity / f"{index:03d}"

        if gpu_faulted() and len(rows) < len(jobs):
            raise RuntimeError("Campaign paused after GPU fault; run doctor --smoke before resuming")

        if active_path.exists():
            active = _read_json(active_path)["job_index"]
            if active == len(rows) and active < len(jobs):
                rows.append(_with_accounting(_recover_job(jobs[active], job_root(active))))
                checkpoint()
            elif active != len(rows) - 1:
                raise ValueError("Campaign active job does not match the frozen job prefix")

        start = len(rows)
        stop = len(jobs) if max_jobs is None else start + max_jobs
        for index, job in enumerate(jobs[start:stop], start):
            root = job_root(index)
            if root.exists():
                raise ValueError("Untracked search directory for next campaign job")
            atomic_json(active_path, {"job_index": index})
            rows.append(_with_accounting(_execute_job(job, config, root)))
            checkpoint()
            if gpu_faulted():
                raise RuntimeError("Campaign paused after GPU fault; run doctor --smoke before resuming")
        return checkpoint()
