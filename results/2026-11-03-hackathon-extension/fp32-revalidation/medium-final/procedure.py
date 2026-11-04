"""Repeat the medium FP32 softmax revalidation after a separate drift diagnostic."""
from dataclasses import asdict
import json
from pathlib import Path
import sys

from metalmind.artifacts.loader import revalidate_artifact
from metalmind.artifacts.store import atomic_json
from metalmind.config import RunConfig
from metalmind.workloads.registry import resolve_reference


def main():
    path, output = map(Path, sys.argv[1:])
    config = RunConfig(domain_mode='assume', warmups=200, seed=202611825,
                       execution_timeout_seconds=420, runs_dir=output/'jobs', artifacts_dir=output/'artifacts')
    result = revalidate_artifact(path, fallback=resolve_reference('softmax'), config=config)
    required = {'original_eager', 'torch_compile_inductor_fullgraph'}
    baseline_sets = []
    for file in sorted(result.run_path.glob('confirmation-*.json')):
        timing = json.loads(file.read_text())['timing']
        if timing:
            baseline_sets.append({item['name'] for item in timing['comparisons']})
    complete = bool(baseline_sets) and all(required.issubset(names) for names in baseline_sets)
    if result.outcome=='promoted' and not complete:
        raise RuntimeError('Missing original verified baseline in a promoted result')
    row = {**asdict(result), 'study':'extension', 'source_artifact':path.name,
           'workload_id':'softmax:medium', 'shape':[256,1024], 'model_requests':0,
           'attempt':3, 'required_original_baselines':sorted(required), 'original_baselines_complete':complete}
    row['run_path'] = str(result.run_path.relative_to(output))
    row['artifact_path'] = str(result.artifact_path.relative_to(output)) if result.artifact_path else None
    atomic_json(output/'summary.json', {'model_requests':0,'rows':[row]})
    print(json.dumps(row), flush=True)


if __name__=='__main__':
    main()
