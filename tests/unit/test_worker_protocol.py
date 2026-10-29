from metalmind.contracts import EvaluationRequest, CandidateSpec, CandidateProposal
from metalmind.runtime.worker import evaluate
from metalmind.runtime.abi import signature_for


def test_already_cancelled_evaluation_does_not_spawn(sample_workload):
    import threading
    cancelled=threading.Event();cancelled.set()
    candidate=CandidateSpec(candidate_id="cancel",workload_hash=signature_for(sample_workload),
        proposal=CandidateProposal(source="invalid",threads=(6,1,1),group_size=(2,1,1),hypothesis="test"))
    request=EvaluationRequest(candidate=candidate,workload=sample_workload)
    assert evaluate(request,cancel_event=cancelled).status=="cancelled"
