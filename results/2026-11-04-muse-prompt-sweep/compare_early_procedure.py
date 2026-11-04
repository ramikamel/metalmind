"""Measure the first two already-confirmed candidates while the sweep continues."""
import json
from pathlib import Path
import runpy
import sys
import time

from metalmind.artifacts.store import atomic_json

def main():
    root=Path(sys.argv[1])
    methods=runpy.run_path(str(root/'compare_procedure.py'))
    prior=methods['prior_artifacts']()
    rows=[]
    for item in json.loads((root/'summary.json').read_text())['rows']:
        key=(item['family'],item['case'],item['dtype'])
        if key not in (('softmax','small','float16'),('rmsnorm','small','float16')):
            continue
        assert item['outcome']=='promoted'
        rows.append({'prior_artifact':str(prior[key]), 'new_artifact':item['artifact_path'],
                     'workload':key,'comparisons':[]})
    assert len(rows)==2
    for session in range(2):
        if session:
            time.sleep(60)
        for index,row in enumerate(rows):
            observation=methods['compare'](row,root,202611600+2*index+session)
            row['comparisons'].append(observation)
            print(row['workload'],session+1,observation.get('summary',observation.get('reason')),flush=True)
    atomic_json(root/'comparison-early.json',{'rows':rows,'model_requests':0})

if __name__=='__main__':
    main()
