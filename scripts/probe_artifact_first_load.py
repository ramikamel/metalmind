"""Measure one offline artifact load and its Metal compile call in this process."""
import argparse
import json
from pathlib import Path
from time import perf_counter_ns
from unittest.mock import patch

from metalmind.artifacts.export import ArtifactManifest
from metalmind.artifacts.loader import load_artifact
from metalmind.runtime.metal import MetalBackend
from metalmind.workloads.registry import resolve_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    manifest = ArtifactManifest.model_validate_json((args.artifact / "manifest.json").read_text())
    if manifest.workload.reference_id != manifest.workload.family:
        parser.error("This probe requires a registered reference")
    fallback = resolve_reference(manifest.workload.reference_id)
    compile_times = []
    compile_kernel = MetalBackend.compile

    def timed_compile(backend, *params, **kwargs):
        started = perf_counter_ns()
        kernel = compile_kernel(backend, *params, **kwargs)
        compile_times.append((perf_counter_ns() - started) / 1e6)
        return kernel

    with patch.object(MetalBackend, "compile", timed_compile):
        started = perf_counter_ns()
        loaded = load_artifact(args.artifact, fallback=fallback)
        load_ms = (perf_counter_ns() - started) / 1e6
    if loaded.optimized is None or len(compile_times) != 1:
        parser.error("No optimized artifact loaded: " + loaded.reason)
    print(json.dumps({
        "artifact_id": manifest.artifact_id,
        "environment_fingerprint": manifest.environment_fingerprint,
        "domain_mode": manifest.domain_mode,
        "load_ms": load_ms,
        "metal_compile_ms": compile_times[0],
    }, allow_nan=False))


if __name__ == "__main__":
    main()
