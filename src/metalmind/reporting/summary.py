"""One summary derived from saved evidence, including rejected candidates."""
from collections import Counter
import json
from pathlib import Path
from metalmind.artifacts.store import ArtifactStore

def build_run_summary(run_path):
    store=ArtifactStore(run_path);events=store.read_events();manifest=store.read_manifest()
    counts=Counter(e['result']['status'] for e in events if e['event']=='evaluation')
    result_path=Path(run_path)/'result.json'
    result=json.loads(result_path.read_text()) if result_path.exists() else None
    candidates=[]
    for p in sorted((Path(run_path)/'candidates').glob('*.metal')):
        candidates.append({'candidate_id':p.stem,'source':p.read_text()})
    private_path=Path(run_path)/'private-verification.json'
    private=json.loads(private_path.read_text()) if private_path.exists() else None
    confirmations=[json.loads(p.read_text()) for p in sorted(Path(run_path).glob('confirmation-*.json'))]
    stages={
        'numerically_valid':any(e['result'].get('verification',{}).get('passed',False)
            for e in events if e['event']=='evaluation' and e['result'].get('verification')),
        'search_fast':any(a.get('score',0)>=1.05 for a in manifest.get('archive',[])),
        'final_verified':bool(private and (private.get('verification') or {}).get('passed')),
        'confirmation_sessions':sum(c.get('status')=='valid' and bool(c.get('timing')) for c in confirmations),
        'promoted':bool(result and result.get('outcome')=='promoted'),
    }
    return {'manifest':manifest,'result':result,'stages':stages,'candidate_counts':dict(counts),
        'final_evidence':{'private_verification':private,'confirmations':confirmations},
        'events':events,'candidates':candidates}
