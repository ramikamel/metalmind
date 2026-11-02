import json
import pytest
from metalmind.agent.meta import MetaProvider,ProviderError
from metalmind.agent.provider import ProposalContext
from metalmind.config import RunConfig


def test_valid_and_malformed_model_outputs(sample_workload):
    proposal={"source":"kernel void metalmind_kernel(){}","threads":[6,1,1],"group_size":[2,1,1],"hypothesis":"fuse"}
    def transport(body):
        assert body['model']=='muse-wonder-1.0'
        return {"model":"muse-wonder-1.0","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":json.dumps(proposal)}]}],"usage":{"output_tokens":100}}
    provider=MetaProvider(RunConfig(),transport=transport)
    response=provider.propose(ProposalContext(workload=sample_workload))
    assert response.proposal.source==proposal['source']
    provider=MetaProvider(RunConfig(),transport=lambda body:{"output":[],"status":"incomplete"})
    assert provider.propose(ProposalContext(workload=sample_workload)).error


def test_auth_error_is_not_retried(sample_workload):
    calls=[]
    def transport(body):
        calls.append(1);raise ProviderError("unauthorized",status=401)
    with pytest.raises(ProviderError):
        MetaProvider(RunConfig(),transport=transport).propose(ProposalContext(workload=sample_workload))
    assert len(calls)==1


@pytest.mark.parametrize("source",['#include /* comment */ "/private/file"\nkernel void metalmind_kernel() {}', '#include <metal_stdlib>\n#pragma clang diagnostic ignored "x"'])
def test_obfuscated_or_extra_preprocessor_directives_are_rejected(source):
    from metalmind.agent.candidates import validate_proposal
    with pytest.raises(ValueError):
        validate_proposal(json.dumps({'source':source,'threads':[6,1,1],'group_size':[2,1,1],'hypothesis':'test'}))
