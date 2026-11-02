from metalmind.contracts import Contract,WorkloadSpec,CandidateProposal
from pydantic import Field

class ProposalContext(Contract):
    workload: WorkloadSpec
    incumbent_source: str | None = None
    feedback: tuple[str,...] = ()
    remaining_requests: int = Field(default=60,ge=1)
    remaining_candidates: int = Field(default=30,ge=1)
    max_output_tokens: int = Field(default=4096,ge=1)
    timeout_seconds: float = Field(default=120,gt=0)

class UsageRecord(Contract):
    input_tokens: int | None = Field(default=None,ge=0)
    output_tokens: int | None = Field(default=None,ge=0)
    reasoning_tokens: int | None = Field(default=None,ge=0)

class ProposalResponse(Contract):
    proposal: CandidateProposal | None = None
    error: str | None = None
    model: str = "unknown"
    usage: UsageRecord = UsageRecord()
    requests: int = Field(default=1,ge=1)

class ReplayProvider:
    def __init__(self,responses): self.responses=iter(responses)
    def propose(self,context): return next(self.responses)
