"""Compare qualifying sweep winners with existing confirmed artifacts."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from metalmind.artifacts.export import _hardware_profile, verify_artifact_evidence
from metalmind.artifacts.loader import load_artifact
from metalmind.artifacts.store import atomic_json
from metalmind.benchmarking.statistics import summarize_pairs
from metalmind.benchmarking.timer import BenchmarkUnstable, measure_pair
from metalmind.runtime.environment import probe_environment
from metalmind.runtime.lock import gpu_lock
from metalmind.verification.numerics import compare_outputs
from metalmind.verification.policy import policy_for
from metalmind.workloads.fixtures import make_public_fixtures
from metalmind.workloads.registry import resolve_reference

def read(path):
    return json.loads(path.read_text())

def prior_artifacts():
    root = Path('results')
    fp16 = root/'2026-11-03-assumed-fp16-pilot'
    old = {(r['family'], r['case'], 'float16'):fp16/'raw'/r['artifact_path']
           for r in read(fp16/'summary.json')['rows'] if r['outcome']=='promoted'}
    for directory, name in (
        (root/'2026-11-03-fp16-confirmation-recheck', 'rows'),
        (root/'2026-11-03-hackathon-extension/short-block-recheck', 'attempts'),
    ):
        for row in read(directory/'summary.json')[name]:
            if row['outcome']=='promoted':
                old[row['family'],row['case'],'float16']=directory/row['artifact_path']
    repair=root/'2026-11-04-muse-prompt-ab/focused-repair'
    old['softmax','medium','float16']=repair/read(repair/'summary.json')['artifact_path']
    for directory in (
        root/'2026-11-03-hackathon-extension/fp32-revalidation',
        root/'2026-11-03-hackathon-extension/fp32-revalidation/medium-final',
    ):
        for row in read(directory/'summary.json')['rows']:
            if row['outcome']=='promoted':
                family,case=row['workload_id'].split(':')
                old[family,case,'float32']=directory/row['artifact_path']
    assert len(old)==13
    return old

def artifact_scores(path):
    manifest=verify_artifact_evidence(path)
    evidence=read(path/'evidence.json')
    ratios=[]
    baselines=[]
    for report in evidence['confirmations']:
        item=min(report['comparisons'],key=lambda c:c['summary']['speedup'])
        ratios.append(item['summary']['speedup'])
        baselines.append(item['name'])
    return manifest,ratios,baselines

def select(sweep):
    existing=prior_artifacts()
    summary=read(sweep/'summary.json')
    assert len(summary['rows'])==18
    new={tuple(row[k] for k in ('family','case','dtype')):(sweep/row['artifact_path'])
         for row in summary['rows'] if row['outcome']=='promoted'}
    for row in read(sweep/'rechecks/summary.json')['attempts']:
        if row['outcome']=='promoted':
            key=tuple(row[k] for k in ('family','case','dtype'))
            assert key not in new
            new[key]=sweep/'rechecks'/row['artifact_path']
    rows=[]
    for key, path in new.items():
        new_manifest,new_ratios,new_baselines=artifact_scores(path)
        row={'family':key[0],'case':key[1],'dtype':key[2],'new_artifact':str(path.relative_to(sweep)),
             'new_ratios':new_ratios,'new_baselines':new_baselines,'comparisons':[]}
        if key not in existing:
            row['decision']='first_confirmed'
        else:
            old_path=existing[key]
            old_manifest,old_ratios,old_baselines=artifact_scores(old_path)
            assert new_manifest.workload==old_manifest.workload and new_manifest.domain_mode==old_manifest.domain_mode=='assume'
            row.update({'prior_artifact':str(old_path),'prior_ratios':old_ratios,'prior_baselines':old_baselines})
            if min(new_ratios)>=1.02*min(old_ratios):
                row['decision']='head_to_head'
            else:
                row['decision']='retained'
        rows.append(row)
    return rows

def compare(row,root,seed):
    old_path=Path(row['prior_artifact'])
    new_path=root/row['new_artifact']
    workload=verify_artifact_evidence(old_path).workload
    reference=resolve_reference(workload.reference_id)
    with gpu_lock(timeout=70),torch.inference_mode():
        prior=load_artifact(old_path,fallback=reference)
        latest=load_artifact(new_path,fallback=reference)
        if prior.optimized is None or latest.optimized is None:
            raise RuntimeError(f"Unable to load artifacts: {prior.reason}; {latest.reason}")
        fixtures=[tuple(t.to("mps") for t in f.inputs) for f in make_public_fixtures(workload)]
        for inputs in fixtures:
            expected=reference(*inputs)
            for kernel in (prior,latest):
                assert compare_outputs(kernel(*inputs),expected,policy_for(workload)).passed
        inputs=fixtures[0]
        try:
            series=measure_pair(lambda:prior(*inputs),lambda:latest(*inputs),pairs=30,warmups=150,seed=seed)
        except BenchmarkUnstable as exc:
            return {'status':'inconclusive','reason':str(exc)}
        result=summarize_pairs(series.baseline_ms,series.candidate_ms,seed=seed)
        return {'status':'valid','summary':result.model_dump(mode='json'),
                'series':series.model_dump(mode='json'),'seed':seed,'public_fixtures':len(fixtures)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sweep',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    rows=select(args.sweep)
    result={'rows':rows,'model_requests':0,'hardware':_hardware_profile(),
            'environment':asdict(probe_environment())}
    atomic_json(args.output,result)
    candidates=[r for r in rows if r['decision']=='head_to_head']
    cached_path=args.sweep/'comparison-early.json'
    if cached_path.exists():
        cached=read(cached_path)['rows']
        for row in candidates:
            match=next((prior for prior in cached if prior['new_artifact']==row['new_artifact']
                        and prior['prior_artifact']==row['prior_artifact']),None)
            if match and len(match['comparisons'])==2:
                row['comparisons']=match['comparisons']
    pending=[(i,r) for i,r in enumerate(candidates) if not r['comparisons']]
    for session in range(2):
        if not pending:
            break
        if session:
            time.sleep(60)
        for index,row in pending:
            observation=compare(row,args.sweep,202611600+2*index+session)
            row['comparisons'].append(observation)
            print(row['dtype'],row['family'],row['case'],session+1,observation.get('summary',observation.get('reason')),flush=True)
            atomic_json(args.output,result)
    for row in candidates:
        row['decision']='improved' if all(
            item['status']=='valid' and item['summary']['speedup']>=1.02 and item['summary']['lower']>1
            for item in row['comparisons']
        ) else 'retained'
    atomic_json(args.output,result)

if __name__=='__main__':
    main()
