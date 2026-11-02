import json
from pathlib import Path
import pytest
import torch
from metalmind.agent.controller import SearchController
from metalmind.agent.provider import ProposalResponse,ReplayProvider,UsageRecord
from metalmind.config import RunConfig
from metalmind.contracts import CandidateProposal,EvaluationResult,VerificationSummary,TimingSeries,TimingReport,BaselineTiming
from metalmind.benchmarking.statistics import summarize_pairs
from metalmind.artifacts.export import export_artifact,verify_artifact_evidence
from metalmind.agent.candidates import make_candidate
from metalmind.workloads.capture import normalize_graph


def cpu_transfer(x):
    return torch.softmax(x.to(dtype=torch.float32,device='cpu'),dim=-1).to(x.dtype)


def test_capture_never_discards_a_device_transfer():
    with pytest.raises(ValueError): normalize_graph(cpu_transfer)


def test_crashed_request_is_reserved_before_dispatch(sample_workload,tmp_path):
    class Crash:
        def propose(self,context): raise SystemExit('simulated crash during billable call')
    cfg=RunConfig(max_requests=1,total_output_tokens=4096,runs_dir=tmp_path)
    with pytest.raises(SystemExit): SearchController(provider=Crash()).run(sample_workload,cfg)
    manifest=json.loads(next(tmp_path.glob('*/manifest.json')).read_text())
    assert manifest['requests']==1 and manifest['output_tokens']==4096
    summary=SearchController(provider=ReplayProvider([])).resume(next(tmp_path.iterdir()))
    assert summary.requests==1


def test_retry_allowance_is_bounded_by_unspent_tokens(sample_workload,tmp_path):
    contexts=[]
    class Provider:
        def propose(self,context):
            contexts.append(context)
            return ProposalResponse(error='no proposal',usage=UsageRecord(output_tokens=4096))
    cfg=RunConfig(max_requests=10,total_output_tokens=4096,runs_dir=tmp_path)
    result=SearchController(provider=Provider()).run(sample_workload,cfg)
    assert len(contexts)==1 and contexts[0].remaining_requests==1
    assert result.requests==1


def test_device_fault_stops_controller(sample_workload,tmp_path):
    proposal=CandidateProposal(source='kernel void metalmind_kernel(){}',threads=(6,1,1),group_size=(2,1,1),hypothesis='test')
    cfg=RunConfig(max_requests=2,runs_dir=tmp_path)
    responses=ReplayProvider([ProposalResponse(proposal=proposal),ProposalResponse(proposal=proposal)])
    def evaluator(request,**kwargs): return EvaluationResult(status='runtime_error',device_fault=True,reason='GPU crashed')
    result=SearchController(provider=responses,evaluator=evaluator).run(sample_workload,cfg)
    assert result.requests==1 and result.outcome=='failed'


def _reports(*,dishonest=False):
    raw_baseline=(.5 if dishonest else 1.1,)*30
    series=TimingSeries(baseline_ms=raw_baseline,candidate_ms=(1.,)*30,repetitions=1,order=('baseline_first',)*30)
    claimed=summarize_pairs((1.1,)*30,(1.,)*30)
    report=TimingReport(comparisons=(BaselineTiming(name='original_eager',series=series,summary=claimed),))
    return [report,report]


def test_mutated_launch_domain_workload_or_abi_invalidates_artifact(sample_workload,tmp_path,monkeypatch):
    import metalmind.artifacts.export as module
    monkeypatch.setattr(module,'environment_fingerprint',lambda:'unit-environment')
    candidate=make_candidate(CandidateProposal(source='kernel void metalmind_kernel(){}',threads=(6,1,1),group_size=(2,1,1),hypothesis='test'),sample_workload)
    folder=export_artifact(candidate,sample_workload,VerificationSummary(passed=True,cases=22),_reports(),root=tmp_path,domain_mode='checked')
    p=folder/'manifest.json';original=json.loads(p.read_text())
    for mutation in ('launch','domain','abi','workload'):
        data=json.loads(json.dumps(original))
        if mutation=='launch': data['candidate']['proposal']['threads']=[99999999,1,1]
        if mutation=='domain': data['domain_mode']='assume'
        if mutation=='abi': data['abi_version']=999
        if mutation=='workload': data['workload']['epsilon']=.25
        p.write_text(json.dumps(data))
        with pytest.raises(ValueError): verify_artifact_evidence(folder)
    p.write_text(json.dumps(original))
    verify_artifact_evidence(folder)


def test_raw_samples_and_case_count_must_support_promotion(sample_workload,tmp_path,monkeypatch):
    import metalmind.artifacts.export as module
    monkeypatch.setattr(module,'environment_fingerprint',lambda:'unit-environment')
    candidate=make_candidate(CandidateProposal(source='kernel void metalmind_kernel(){}',threads=(6,1,1),group_size=(2,1,1),hypothesis='test'),sample_workload)
    with pytest.raises(ValueError):
        export_artifact(candidate,sample_workload,VerificationSummary(passed=True,cases=0),_reports(),root=tmp_path,domain_mode='checked')
    with pytest.raises(ValueError):
        export_artifact(candidate,sample_workload,VerificationSummary(passed=True,cases=22),_reports(dishonest=True),root=tmp_path,domain_mode='checked')


def test_baseline_sets_must_match_and_expired_export_never_publishes(sample_workload,tmp_path,monkeypatch):
    import time
    import metalmind.artifacts.export as module
    monkeypatch.setattr(module,'environment_fingerprint',lambda:'unit-environment')
    candidate=make_candidate(CandidateProposal(source='kernel void metalmind_kernel(){}',threads=(6,1,1),group_size=(2,1,1),hypothesis='test'),sample_workload)
    reports=_reports()
    reports[1]=reports[1].model_copy(update={'comparisons':(reports[1].comparisons[0].model_copy(update={'name':'different_baseline'}),)})
    with pytest.raises(ValueError): export_artifact(candidate,sample_workload,VerificationSummary(passed=True,cases=22),reports,root=tmp_path,domain_mode='checked')
    with pytest.raises(TimeoutError): export_artifact(candidate,sample_workload,VerificationSummary(passed=True,cases=22),_reports(),root=tmp_path,domain_mode='checked',deadline=time.monotonic()-1)
    assert not list(tmp_path.iterdir())


def test_finalized_run_cannot_spend_requests_on_resume(sample_workload,tmp_path):
    from metalmind.agent.controller import RunSummary
    from metalmind.artifacts.export import finalize_run
    summary=RunSummary(workload=sample_workload,config=RunConfig(),run_path=tmp_path,outcome='budget_exhausted')
    finalize_run(summary)
    assert json.loads((tmp_path/'manifest.json').read_text())['phase']=='terminal'
    with pytest.raises(ValueError): SearchController(provider=ReplayProvider([])).resume(tmp_path)


def test_malformed_evidence_uses_fallback(tmp_path,monkeypatch):
    from metalmind.artifacts.loader import load_artifact
    import metalmind.artifacts.loader as module
    def broken(path): raise KeyError('verification')
    monkeypatch.setattr(module,'verify_artifact_evidence',broken)
    assert load_artifact(tmp_path,fallback=lambda x:x+1)(4)==5


def test_existing_gpu_fault_blocks_new_model_requests(sample_workload,tmp_path,monkeypatch):
    import metalmind.runtime.worker as worker
    marker=tmp_path/'gpu-fault';marker.write_text('fault')
    monkeypatch.setattr(worker,'FAULT_PATH',marker)
    class MustNotCall:
        def propose(self,context): raise AssertionError('model request after GPU fault')
    result=SearchController(provider=MustNotCall()).run(sample_workload,RunConfig(runs_dir=tmp_path/'runs'))
    assert result.requests==0 and result.outcome=='failed'


def test_unapproved_headers_cannot_be_exported(sample_workload,tmp_path,monkeypatch):
    import metalmind.artifacts.export as module
    monkeypatch.setattr(module,'environment_fingerprint',lambda:'unit-environment')
    proposal=CandidateProposal(source='#include /*comment*/ "/private/file"\nkernel void metalmind_kernel(){}',threads=(6,1,1),group_size=(2,1,1),hypothesis='test')
    candidate=make_candidate(proposal,sample_workload)
    with pytest.raises(ValueError):
        export_artifact(candidate,sample_workload,VerificationSummary(passed=True,cases=22),_reports(),root=tmp_path,domain_mode='checked')


def test_artifact_fingerprint_distinguishes_gpu_core_variants(monkeypatch):
    import metalmind.artifacts.export as module
    import subprocess
    monkeypatch.setattr(subprocess,'check_output',lambda *args,**kwargs:'Apple test chip')
    monkeypatch.setattr(module,'_hardware_profile',lambda:{'model':'Apple M3 Pro','cores':'14'},raising=False)
    first=module.environment_fingerprint()
    monkeypatch.setattr(module,'_hardware_profile',lambda:{'model':'Apple M3 Pro','cores':'18'},raising=False)
    assert module.environment_fingerprint()!=first
