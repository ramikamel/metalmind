from metalmind.agent.controller import SearchController
from metalmind.agent.provider import ReplayProvider
from metalmind.config import RunConfig


def test_no_request_dispatched_when_output_reservation_exceeds_budget(sample_workload,tmp_path):
    config=RunConfig(total_output_tokens=1,runs_dir=tmp_path)
    summary=SearchController(provider=ReplayProvider([])).run(sample_workload,config)
    assert summary.requests==0
    assert summary.outcome=="budget_exhausted"
