"""Observe baseline stability around paired windows for one saved candidate."""
from dataclasses import asdict
import json
from pathlib import Path
import sys

import torch
from metalmind.artifacts.export import verify_artifact_evidence
from metalmind.artifacts.loader import load_artifact
from metalmind.artifacts.store import atomic_json
from metalmind.benchmarking.baselines import collect_baselines
from metalmind.benchmarking import timer
from metalmind.runtime.lock import gpu_lock
from metalmind.workloads.fixtures import make_public_fixtures
from metalmind.workloads.registry import resolve_reference


def main():
    artifact, output = map(Path, sys.argv[1:])
    data = []
    manifest = verify_artifact_evidence(artifact)
    workload = manifest.workload
    raw_block = timer._block
    with gpu_lock(), torch.inference_mode():
        reference = resolve_reference(workload.reference_id)
        optimized = load_artifact(artifact, fallback=reference)
        assert optimized.optimized is not None, optimized.reason
        inputs = tuple(t.to('mps') for t in make_public_fixtures(workload)[0].inputs)
        baselines, unavailable = collect_baselines(workload)
        assert not unavailable, unavailable
        for index in range(2):
            for name, reference in baselines:
                def baseline():
                    return reference(*inputs)

                def candidate():
                    return optimized(*inputs)

                blocks = []

                def observe(fn, repetitions):
                    elapsed = raw_block(fn, repetitions)
                    blocks.append({'arm':'baseline' if fn is baseline else 'candidate',
                                   'repetitions':repetitions,'ms':elapsed})
                    return elapsed

                timer._block = observe
                try:
                    result = timer.measure_pair(baseline, candidate, pairs=30, warmups=100, seed=1729+index)
                    outcome = {'status':'valid','series':result.model_dump(mode='json')}
                except timer.BenchmarkUnstable as exc:
                    outcome = {'status':'unstable','reason':str(exc)}
                finally:
                    timer._block = raw_block
                data.append({'window':index+1,'baseline':name,'outcome':outcome,'blocks':blocks})
                atomic_json(output, {'windows':data,'model_requests':0})
                print(index+1, name, outcome['status'], flush=True)


if __name__=='__main__':
    main()
