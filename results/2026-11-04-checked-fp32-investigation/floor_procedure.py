"""Measure how the best available PyTorch baseline compares with the guard alone."""
import argparse
from dataclasses import asdict
from pathlib import Path
import runpy

import torch

from metalmind.artifacts.export import _hardware_profile
from metalmind.artifacts.store import atomic_json
from metalmind.benchmarking.baselines import collect_baselines
from metalmind.runtime.environment import probe_environment
from metalmind.runtime.lock import gpu_lock
from metalmind.workloads.fixtures import make_public_fixtures
from metalmind.workloads.registry import get_workload

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    helpers=runpy.run_path(str(Path(__file__).with_name('procedure.py')))
    results=[]
    for family in ('bias_gelu','rmsnorm','softmax'):
        for size in ('small','medium','large'):
            workload=get_workload(family,size,'float32')
            with gpu_lock(timeout=70), torch.inference_mode():
                check=helpers['make_check'](workload)
                inputs=tuple(t.to('mps') for t in make_public_fixtures(workload)[0].inputs)
                assert check(inputs)
                options,unavailable=collect_baselines(workload)
                comparisons={name:helpers['pair'](lambda:fn(*inputs),lambda:check(inputs),11347)
                             for name,fn in options}
            results.append({'workload':workload.workload_id,'comparisons':comparisons,
                            'unavailable':unavailable})
            atomic_json(args.output,{'rows':results,'model_requests':0,'hardware':_hardware_profile(),
                                     'environment':asdict(probe_environment())})
            print(workload.workload_id, {name: (row['baseline_ms'],row['candidate_ms'])
                  if row['status']=='valid' else row['reason'] for name,row in comparisons.items()}, flush=True)

if __name__=='__main__':
    main()
