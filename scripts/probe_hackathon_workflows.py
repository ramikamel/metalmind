"""Measure a same-kernel FP32 boundary and a complete FP16 attention-style block."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from metalmind.artifacts.export import _hardware_profile, environment_fingerprint, verify_artifact_evidence
from metalmind.artifacts.loader import load_artifact
from metalmind.artifacts.store import atomic_json
from metalmind.benchmarking.baselines import collect_baselines
from metalmind.benchmarking.statistics import summarize_pairs
from metalmind.benchmarking.timer import BenchmarkUnstable, measure_pair
from metalmind.demonstration.provenance import runtime_source_info
from metalmind.runtime.abi import matches_signature
from metalmind.runtime.environment import probe_environment
from metalmind.runtime.execution import build_execution_callable
from metalmind.runtime.lock import gpu_lock
from metalmind.runtime.metal import MetalBackend
from metalmind.verification.numerics import compare_outputs, inputs_in_domain
from metalmind.verification.policy import policy_for
from metalmind.workloads.fixtures import make_public_fixtures
from metalmind.workloads.registry import resolve_reference

FP32 = Path("results/2026-11-02-fp32-pilot/artifacts/db066f6dfe2e74cefc0c335852229dd2ed6a9058518279eabb7b3a28342caa87")
FP16 = Path("results/2026-11-03-assumed-fp16-pilot")
HEAD_WIDTH = 64


def paired(reference, candidate, *, seed):
    try:
        series = measure_pair(reference, candidate, pairs=30, warmups=100, seed=seed)
    except BenchmarkUnstable as exc:
        return {"status": "inconclusive", "reason": str(exc)}
    summary = summarize_pairs(series.baseline_ms, series.candidate_ms, seed=seed)
    return {"status": "valid", "seed": seed, "series": series.model_dump(mode="json"),
            "summary": summary.model_dump(mode="json")}


def fp32_boundary():
    artifact = verify_artifact_evidence(FP32)
    if artifact.environment_fingerprint != environment_fingerprint():
        raise ValueError("FP32 artifact environment differs from the benchmark machine")
    workload = artifact.workload
    reference = resolve_reference(workload.reference_id)
    fallback_count = 0

    def fallback(*inputs):
        nonlocal fallback_count
        fallback_count += 1
        return reference(*inputs)
    kernel = MetalBackend().compile(artifact.candidate.proposal.source, entry_point="metalmind_kernel")
    wrappers = {}
    for mode in ("checked", "assume"):
        fn = build_execution_callable(workload, kernel, fallback=fallback, domain_mode=mode)
        fn.proposal = artifact.candidate.proposal
        wrappers[mode] = fn
    fixtures = [tuple(t.to("mps") for t in f.inputs) for f in make_public_fixtures(workload)]
    for inputs in fixtures:
        assert inputs_in_domain(workload, inputs)
        for fn in wrappers.values():
            assert compare_outputs(fn(*inputs), reference(*inputs), policy_for(workload)).passed
    assert fallback_count == 0
    x, bias = fixtures[0]
    out_of_range = x.clone()
    out_of_range[0, 0] = 11
    torch.testing.assert_close(wrappers["checked"](out_of_range, bias), reference(out_of_range, bias), atol=0, rtol=0)
    assert fallback_count == 1
    baselines, unavailable = collect_baselines(workload)
    if "torch_compile_inductor_fullgraph" not in {name for name, _ in baselines}:
        raise RuntimeError("The archived compiled FP32 baseline is required")
    return artifact.artifact_id, wrappers, fixtures[0], baselines, unavailable


def attention_inputs(seed):
    generator = torch.Generator().manual_seed(seed)
    # Dyadic inputs keep every nonzero score at or above 1/2048 in magnitude.
    shapes = ((256, HEAD_WIDTH), (1024, HEAD_WIDTH), (1024, HEAD_WIDTH))
    return tuple((torch.randint(-8, 9, shape, generator=generator).half() / 16).to("mps")
                 for shape in shapes)


def attention_scores(q, k):
    return (q @ k.transpose(-2, -1)) * (HEAD_WIDTH ** -.5)


def attention_baselines(path):
    manifest = verify_artifact_evidence(path)
    if manifest.domain_mode != "assume":
        raise ValueError("Attention diagnostic requires an assumed-domain softmax artifact")
    original_softmax = resolve_reference(manifest.workload.reference_id)
    optimized_softmax = load_artifact(path, fallback=original_softmax)
    if optimized_softmax.optimized is None:
        raise RuntimeError(optimized_softmax.reason)

    def original(q, k, v):
        return original_softmax(attention_scores(q, k)) @ v

    def candidate(q, k, v):
        return optimized_softmax(attention_scores(q, k)) @ v

    def builtin(q, k, v):
        return F.scaled_dot_product_attention(q[None, None], k[None, None], v[None, None])[0, 0]

    fixtures = [attention_inputs(seed) for seed in range(91, 99)]
    policy = policy_for(manifest.workload)
    for q, k, v in fixtures:
        scores = (attention_scores(q, k),)
        assert matches_signature(manifest.workload, scores)
        assert inputs_in_domain(manifest.workload, scores)
        assert compare_outputs(candidate(q, k, v), original(q, k, v), policy).passed
    options = [("eager_split_attention", original), ("pytorch_scaled_dot_product_attention", builtin)]
    unavailable = []
    try:
        options.append(("inductor_fullgraph_split_attention", torch.compile(original, fullgraph=True)))
    except Exception as exc:
        unavailable.append({"baseline": "inductor_fullgraph_split_attention", "reason": str(exc)[:300]})
    valid = []
    for name, fn in options:
        try:
            for inputs in fixtures:
                actual = fn(*inputs)
                if actual.device.type != "mps" or not compare_outputs(actual, original(*inputs), policy).passed:
                    raise ValueError("Whole output failed numerical verification")
            valid.append((name, fn))
        except Exception as exc:
            unavailable.append({"baseline": name, "reason": str(exc)[:300]})
    if not valid or valid[0][0] != "eager_split_attention":
        raise RuntimeError("Original eager attention baseline is invalid")
    return manifest.artifact_id, candidate, fixtures[0], valid, unavailable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    fp16_summary = json.loads((FP16 / "summary.json").read_text())
    row = next(row for row in fp16_summary["rows"] if (row["family"], row["case"]) == ("softmax", "medium"))
    fp16_path = FP16 / "raw" / row["artifact_path"]
    result = {"model_requests": 0, "hardware": _hardware_profile(), "sessions": [],
              "environment": asdict(probe_environment()), "runtime_source": runtime_source_info(),
              "protocol": {"pairs": 30, "warmups": 100, "cooldown_seconds": 60,
                           "calibration": "smaller of the two calibrated repetition counts"}}
    with gpu_lock(), torch.inference_mode():
        for session in (1, 2):
            if session == 2:
                time.sleep(60)
            guard_id, wrappers, inputs, baselines, missing = fp32_boundary()
            scan = paired(lambda: wrappers["checked"](*inputs), lambda: wrappers["assume"](*inputs), seed=7100+session)
            fp32 = {}
            modes = ("checked", "assume") if session == 1 else ("assume", "checked")
            for mode in modes:
                fp32[mode] = {
                    name: paired(lambda: fn(*inputs), lambda: wrappers[mode](*inputs), seed=7100+session)
                    for name, fn in baselines
                }
            result["sessions"].append({"session": session, "fp32": {
                "artifact_id": guard_id, "checked_divided_by_assume": scan, "comparisons": fp32,
                "unavailable": missing, "outside_domain_fallback": True,
            }})
            atomic_json(args.output, result)

            artifact_id, candidate, attention, attention_base, unavailable = attention_baselines(fp16_path)
            comparisons = {
                name: paired(lambda: fn(*attention), lambda: candidate(*attention), seed=7200+session)
                for name, fn in attention_base
            }
            result["sessions"][-1]["fp16_attention"] = {
                "artifact_id": artifact_id, "comparisons": comparisons, "unavailable": unavailable,
                "whole_output_fixtures_verified": 8, "domain_mode": "assume",
            }
            atomic_json(args.output, result)


if __name__ == "__main__":
    main()
