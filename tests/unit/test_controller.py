from metalmind.agent.controller import SearchController
from metalmind.agent.provider import ReplayProvider, ProposalResponse
from metalmind.contracts import CandidateProposal,EvaluationResult,VerificationSummary
from metalmind.config import RunConfig


def test_feedback_budget_and_duplicate_source_handling(sample_workload,tmp_path):
    proposal=CandidateProposal(source="kernel void metalmind_kernel(){}",threads=(6,1,1),group_size=(2,1,1),hypothesis="first")
    duplicate=proposal.model_copy(update={"hypothesis":"different explanation"})
    provider=ReplayProvider([ProposalResponse(error="malformed"),ProposalResponse(proposal=proposal),ProposalResponse(proposal=duplicate)])
    evaluations=[]
    def evaluator(request,**kwargs):
        evaluations.append(request)
        return EvaluationResult(status="valid",verification=VerificationSummary(passed=True,cases=10))
    config=RunConfig(max_requests=3,max_candidates=3,runs_dir=tmp_path)
    summary=SearchController(provider=provider,evaluator=evaluator).run(sample_workload,config)
    assert summary.requests==3
    assert summary.evaluated==1
    assert len(evaluations)==1
    assert summary.incumbent is None  # No measured candidate can be selected.
    assert (summary.run_path/"manifest.json").is_file()
