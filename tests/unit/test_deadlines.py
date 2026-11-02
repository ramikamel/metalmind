import time
from metalmind.contracts import EvaluationRequest,CandidateSpec,CandidateProposal
from metalmind.runtime.worker import evaluate
from metalmind.runtime.abi import signature_for


def test_absolute_deadline_includes_worker_startup(sample_workload,monkeypatch,tmp_path):
    import metalmind.runtime.worker as worker
    monkeypatch.setattr(worker,'FAULT_PATH',tmp_path/'fault.json',raising=False)
    candidate=CandidateSpec(candidate_id='test',workload_hash=signature_for(sample_workload),proposal=CandidateProposal(source='invalid',threads=(6,1,1),group_size=(2,1,1),hypothesis='test'))
    request=EvaluationRequest(candidate=candidate,workload=sample_workload,total_timeout_seconds=.01)
    start=time.monotonic();result=evaluate(request)
    assert result.status=='timeout'
    assert time.monotonic()-start<3
