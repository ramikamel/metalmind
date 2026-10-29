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
    includes=re.findall(r'\s*#\s*include\s*[<"]([^>"]+)',proposal.source)
    if any(s!="metal_stdlib" for s in includes): raise ValueError("only metal_stdlib may be included")
    if re.search(r"#\s*(?:define|import|include_next)\b",proposal.source):
        raise ValueError("preprocessor macros/imports are outside initial source policy")
    return proposal

def make_candidate(proposal,workload,*,parent_id=None,model="manual"):
    identity=signature_for(workload)+json.dumps(proposal.model_dump(exclude={"hypothesis"}),sort_keys=True)
    return CandidateSpec(candidate_id=hashlib.sha256(identity.encode()).hexdigest(),
        workload_hash=signature_for(workload),proposal=proposal,parent_id=parent_id,model=model)
