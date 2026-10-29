"""Durable bounded search; only numerically valid measured candidates compete."""
import json
from pathlib import Path
import threading
import time
from pydantic import Field
from metalmind.contracts import Contract,WorkloadSpec,CandidateSpec,EvaluationRequest
from metalmind.config import RunConfig
from metalmind.agent.provider import ProposalContext
from metalmind.agent.meta import MetaProvider,ProviderError
from metalmind.agent.candidates import make_candidate
from metalmind.artifacts.store import ArtifactStore
from metalmind.runtime.environment import probe_environment
from metalmind.runtime.worker import evaluate

class RunSummary(Contract):
    workload: WorkloadSpec
    config: RunConfig
    run_path: Path
    outcome: str
    incumbent: CandidateSpec | None = None
    incumbent_score: float | None = None
    requests: int = 0
    evaluated: int = 0
    output_tokens: int = 0

class SearchController:
    def __init__(self,*,provider=None,evaluator=evaluate):
        self.provider=provider;self.evaluator=evaluator;self.cancel_event=threading.Event()

    def cancel(self): self.cancel_event.set()

    def run(self,workload,config,*,resume_path=None):
        environment=probe_environment()
        store=ArtifactStore.create_run(config.runs_dir,config,environment) if resume_path is None else ArtifactStore(resume_path)
        provider=self.provider or MetaProvider(config)
        requests=evaluated=tokens=stagnation=0;seen=set();archive=[];feedback=[]
        elapsed_before=0.
        if resume_path is not None:
            previous=store.read_manifest()
            if previous['workload']!=workload.model_dump(mode="json") or previous['config']!=config.model_dump(mode="json"):
                raise ValueError("resume workload/config mismatch")
            from dataclasses import asdict
            if previous['environment']!=asdict(environment): raise ValueError("resume environment mismatch")
            requests=previous['requests'];evaluated=previous['evaluated'];tokens=previous['output_tokens']
            elapsed_before=previous['elapsed_seconds'];seen=set(previous['seen'])
            archive=[(entry['score'],CandidateSpec.model_validate(entry['candidate'])) for entry in previous['archive']]
        started=time.monotonic();search_seconds=config.wall_seconds-config.finalization_reserve_seconds
        outcome="budget_exhausted"
        def snapshot(phase):
            from dataclasses import asdict
            store.write_manifest({"schema_version":1,"phase":phase,"workload":workload.model_dump(mode="json"),
                "config":config.model_dump(mode="json"),"environment":asdict(environment),
                "requests":requests,"evaluated":evaluated,"output_tokens":tokens,
                "elapsed_seconds":elapsed_before+time.monotonic()-started,"seen":sorted(seen),
                "archive":[{"score":s,"candidate":c.model_dump(mode="json")} for s,c in archive]})
        snapshot("proposing")
        while requests<config.max_requests and evaluated<config.max_candidates:
            remaining=search_seconds-elapsed_before-(time.monotonic()-started)
            if self.cancel_event.is_set(): outcome="cancelled";break
            if remaining<=0 or tokens+config.max_output_tokens>config.total_output_tokens: break
            incumbent=archive[0][1] if archive else None
            context=ProposalContext(workload=workload,incumbent_source=incumbent.proposal.source if incumbent else None,
                feedback=tuple(feedback[-8:]),remaining_requests=config.max_requests-requests,
                remaining_candidates=config.max_candidates-evaluated,max_output_tokens=config.max_output_tokens,
                timeout_seconds=min(config.request_timeout_seconds,remaining))
            try: response=provider.propose(context)
            except ProviderError as exc:
                requests+=exc.requests
                feedback.append(str(exc));store.append_event({"event":"provider_error","reason":str(exc),"requests":exc.requests})
                outcome="failed";break
            except StopIteration: break
            requests+=response.requests
            tokens+=response.usage.output_tokens if response.usage.output_tokens is not None else config.max_output_tokens*response.requests
            store.append_event({"event":"model_response","model":response.model,"usage":response.usage.model_dump(),"requests":response.requests,"error":response.error})
            if response.proposal is None:
                feedback.append("Schema rejection: "+str(response.error));snapshot("proposing");continue
            candidate=make_candidate(response.proposal,workload,parent_id=incumbent.candidate_id if incumbent else None,model=response.model)
            if candidate.candidate_id in seen:
                feedback.append("Duplicate source/launch already evaluated; propose a different implementation.")
                snapshot("proposing");continue
            seen.add(candidate.candidate_id);store.write_source(candidate.candidate_id,candidate.proposal.source)
            remaining=search_seconds-elapsed_before-(time.monotonic()-started)
            if remaining<=0: break
            request=EvaluationRequest(candidate=candidate,workload=workload,measure=True,domain_mode=config.domain_mode,
                seed=config.seed,compile_timeout_seconds=min(config.compile_timeout_seconds,remaining),
                execution_timeout_seconds=min(config.execution_timeout_seconds,max(.01,remaining)))
            result=self.evaluator(request,cancel_event=self.cancel_event);evaluated+=1
            store.append_event({"event":"evaluation","candidate_id":candidate.candidate_id,"result":result.model_dump(mode="json")})
            old=archive[0][0] if archive else 0.
            if result.status=="valid" and result.timing is not None:
                score=min(c.summary.speedup for c in result.timing.comparisons)
                archive.append((score,candidate));archive.sort(key=lambda pair:pair[0],reverse=True);archive=archive[:3]
                feedback.append(f"Verified candidate {candidate.candidate_id[:8]}: minimum public-wrapper speedup {score:.4f}; baseline comparisons "+result.timing.model_dump_json())
                stagnation=0 if score>old*1.01 else stagnation+1
            else:
                feedback.append(f"{result.status}: {result.reason or (result.verification.reason if result.verification else '')}")
                stagnation+=1
            snapshot("proposing")
            if result.status in ("timeout","cancelled"):
                outcome="cancelled" if result.status=="cancelled" else "failed";break
            if stagnation>=config.stagnation_limit: outcome="stagnated";break
        snapshot("search_complete")
        return RunSummary(workload=workload,config=config,run_path=store.path,outcome=outcome,
            incumbent=archive[0][1] if archive else None,incumbent_score=archive[0][0] if archive else None,
            requests=requests,evaluated=evaluated,output_tokens=tokens)

    def resume(self,run_path):
        data=ArtifactStore(run_path).read_manifest()
        if data['phase'] not in ('proposing','search_complete'): raise ValueError("run cannot resume from this phase")
        return self.run(WorkloadSpec.model_validate(data['workload']),RunConfig.model_validate(data['config']),resume_path=run_path)
