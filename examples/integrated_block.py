"""Measure an unchanged matmul plus a confirmed bias/GELU replacement.

Run: python -m examples.integrated_block --artifact artifacts/<id>
Custom-reference artifacts additionally require --function module:name.
This does not call a model. It reports whole-workflow latency, not just GELU.
"""
import argparse
import json
from pathlib import Path

import torch
from metalmind.artifacts.export import verify_artifact_evidence
from metalmind.artifacts.loader import load_artifact
from metalmind.benchmarking.timer import measure_pair
from metalmind.benchmarking.statistics import summarize_pairs
from metalmind.workloads.registry import resolve_reference
from metalmind.verification.numerics import inputs_in_domain


def build_workflow(workload, reference, activation):
    if workload.family != "bias_gelu":
        raise ValueError("This example requires a bias/GELU artifact")
    b, d = workload.inputs[0].shape
    dtype = getattr(torch, workload.inputs[0].dtype)
    # Bounds: each product <= .001, so matmul <= D*.001 (4.096 at D=4096).
    if d > 4096:
        raise ValueError("This example's bounded input construction requires D <= 4096")
    x = torch.empty(b, d, device="mps", dtype=dtype).uniform_(-.1, .1)
    w = torch.empty(d, d, device="mps", dtype=dtype).uniform_(-.01, .01)
    bias = torch.empty(d, device="mps", dtype=dtype).uniform_(-.1, .1)
    with torch.inference_mode():
        if not inputs_in_domain(workload, (x @ w, bias)):
            raise ValueError("Constructed activation inputs do not satisfy the artifact domain")
    def original():
        return reference(x @ w, bias)
    def optimized():
        return activation(x @ w, bias)
    return original, optimized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--function", help="Explicit module:name for a trusted custom reference")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = verify_artifact_evidence(args.artifact)
    expected = manifest.workload.reference_id
    if expected.startswith("python:"):
        if args.function != expected.removeprefix("python:"):
            parser.error("Custom artifacts require their matching --function module:name")
    elif args.function:
        parser.error("Registered artifacts resolve their registered reference automatically")
    reference = resolve_reference(expected)
    activation = load_artifact(args.artifact, fallback=reference)
    if activation.optimized is None:
        parser.error("Artifact is fallback-only: " + activation.reason)
    original, optimized = build_workflow(manifest.workload, reference, activation)
    with torch.inference_mode():
        torch.testing.assert_close(optimized(), original(), atol=1e-5, rtol=1e-4)
    series = measure_pair(original, optimized, pairs=30)
    result = {
        "artifact_id": manifest.artifact_id,
        "domain_mode": manifest.domain_mode,
        "shape": manifest.workload.inputs[0].shape,
        "boundary": "unchanged matmul plus guarded bias/GELU activation",
        "optimized_enabled": True,
        "outputs_match": True,
        "series": series.model_dump(mode="json"),
        "summary": summarize_pairs(series.baseline_ms, series.candidate_ms).model_dump(mode="json"),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
