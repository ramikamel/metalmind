import hashlib
import json
import re
from metalmind.contracts import CandidateProposal,CandidateSpec
from metalmind.runtime.abi import signature_for

def validate_proposal(raw):
    text=raw.strip()
    if text.startswith("```json") and text.endswith("```"): text=text[7:-3].strip()
    proposal=CandidateProposal.model_validate_json(text)
    if len(proposal.source.encode())>65536: raise ValueError("source byte budget exceeded")
    if chr(92) in proposal.source or '??' in proposal.source:
        raise ValueError("source continuations/trigraphs are outside initial policy")
    for line in proposal.source.splitlines():
        if '#' in line and re.fullmatch(r'\s*#\s*include\s*<metal_stdlib>\s*',line) is None:
            raise ValueError("only a direct metal_stdlib include is allowed")
    return proposal

def make_candidate(proposal,workload,*,parent_id=None,model="manual"):
    identity=signature_for(workload)+json.dumps(proposal.model_dump(exclude={"hypothesis"}),sort_keys=True)
    return CandidateSpec(candidate_id=hashlib.sha256(identity.encode()).hexdigest(),
        workload_hash=signature_for(workload),proposal=proposal,parent_id=parent_id,model=model)
