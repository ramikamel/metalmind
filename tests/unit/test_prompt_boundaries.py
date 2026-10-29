from metalmind.agent.provider import ProposalContext
from metalmind.agent.prompts import build_prompt


def test_prompt_contains_contract_but_no_private_fixture_data(sample_workload):
    prompt=build_prompt(ProposalContext(workload=sample_workload))
    assert "metalmind_kernel" in prompt and "bias_gelu" in prompt
    assert "private_seeds" not in prompt and "MODEL_API_KEY" not in prompt
    assert "tanh" in prompt
