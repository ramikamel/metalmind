"""Run current short-block confirmations for the four previously promoted FP32 artifacts."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

from metalmind.artifacts.export import verify_artifact_evidence
from metalmind.artifacts.loader import revalidate_artifact
from metalmind.artifacts.store import atomic_json
from metalmind.config import RunConfig
from metalmind.runtime.worker import gpu_faulted
from metalmind.workloads.registry import resolve_reference


def main():
    search, output = map(Path, sys.argv[1:])
    original = Path('results/2026-11-02-fp32-pilot/artifacts')
    pilot = sorted(original.iterdir())
    new = [Path(row['artifact_path']) for row in json.loads((search/'summary.json').read_text())['rows']
           if row['dtype']=='float32' and row['outcome']=='promoted']
    rows = []
    for index, (study, path) in enumerate([*(('extension', path) for path in new),
                                          *(('pilot', path) for path in pilot)]):
        manifest = verify_artifact_evidence(path)
        workload = manifest.workload
        evidence = json.loads((path/'evidence.json').read_text())
        required_baselines = {item['name'] for item in evidence['confirmations'][0]['comparisons']}
        assert workload.inputs[0].dtype == 'float32' and manifest.domain_mode == 'assume'
        for attempt in range(2):
            config = RunConfig(domain_mode='assume', warmups=100, seed=202611521+4*index+2*attempt,
                               execution_timeout_seconds=420, runs_dir=output/'jobs', artifacts_dir=output/'artifacts')
            print(f'Rechecking {workload.workload_id} from {study}, attempt {attempt+1}', flush=True)
            result = revalidate_artifact(path, fallback=resolve_reference(workload.reference_id), config=config)
            saved = []
            for report in sorted(result.run_path.glob('confirmation-*.json')):
                timing = json.loads(report.read_text()).get('timing')
                if timing:
                    saved.append({item['name'] for item in timing['comparisons']})
            baselines_complete = bool(saved) and all(required_baselines.issubset(names) for names in saved)
            if result.outcome=='promoted' and not baselines_complete:
                raise RuntimeError('Missing original verified baseline in a promoted result')
            row = {**asdict(result), 'study':study, 'source_artifact':path.name,
                   'workload_id':workload.workload_id, 'shape':workload.inputs[0].shape,
                   'model_requests':0, 'attempt':attempt+1, 'required_original_baselines':sorted(required_baselines),
                   'original_baselines_complete':baselines_complete}
            row['run_path'] = str(result.run_path.relative_to(output))
            row['artifact_path'] = str(result.artifact_path.relative_to(output)) if result.artifact_path else None
            rows.append(row)
            atomic_json(output/'summary.json', {'model_requests':0, 'rows':rows})
            print(json.dumps(row), flush=True)
            if gpu_faulted():
                subprocess.run([sys.executable, '-m', 'metalmind.cli', 'doctor', '--smoke', '--json'], check=True)
            if result.outcome!='inconclusive' or result.reason!='Confirmation did not yield stable valid measurements':
                break


if __name__ == '__main__':
    main()
