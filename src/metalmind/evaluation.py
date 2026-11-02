"""Frozen sequential campaigns with explicit denominators and strategy labels."""
import itertools
import math
from pathlib import Path
from typing import Literal
from pydantic import Field
from metalmind.contracts import Contract,CandidateProposal
from metalmind.config import RunConfig
from metalmind.artifacts.export import digest,finalize_run
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

def run_campaign(config,*,output_root=Path('runs/campaigns')):
    frozen=freeze_campaign(config);folder=Path(output_root)/frozen['campaign_id'];folder.mkdir(parents=True,exist_ok=False)
    atomic_json(folder/'campaign.json',frozen);rows=[]
    jobs=list(itertools.product(config.families,config.cases,config.strategies,range(config.trials)))
    import random
    random.Random(config.run_config.seed).shuffle(jobs)
    for family,case,strategy,trial in jobs:
        workload=get_workload(family,case,config.dtype)
        cfg=config.run_config.model_copy(update={'domain_mode':config.domain_mode,'seed':config.run_config.seed+trial})
        if strategy=='one_shot': cfg=cfg.model_copy(update={'max_candidates':1,'max_requests':1})
        if strategy=='template': provider=ReplayProvider(template_proposals(workload))
        elif strategy=='independent': provider=IndependentProvider(MetaProvider(cfg))
        else: provider=None
        try:
            summary=SearchController(provider=provider).run(workload,cfg)
            result=finalize_run(summary)
            speedup=1.0
            if result.artifact_path:
                import json
                evidence=json.loads((result.artifact_path/'evidence.json').read_text())
                speedup=min(c['summary']['speedup'] for s in evidence['confirmations'] for c in s['comparisons'])
            row={'family':family,'case':case,'strategy':strategy,'trial':trial,'outcome':result.outcome,'speedup':speedup,'run_path':str(summary.run_path),'requests':summary.requests}
        except Exception as exc:
            row={'family':family,'case':case,'strategy':strategy,'trial':trial,'outcome':'failed','speedup':1.0,'reason':str(exc)[:500]}
        rows.append(row);atomic_json(folder/'summary.json',summarize_campaign(rows))
        from metalmind.runtime.worker import gpu_faulted
        if gpu_faulted(): raise RuntimeError('Campaign stopped after GPU fault; run doctor --smoke before resuming GPU work')
    return summarize_campaign(rows)
