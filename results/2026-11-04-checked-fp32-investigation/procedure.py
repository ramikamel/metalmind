"""Probe a single Metal domain scan against the existing checked FP32 path."""
import argparse
from dataclasses import asdict
import json
from math import prod
from pathlib import Path
import statistics
import struct

import torch

from metalmind.artifacts.export import _hardware_profile, verify_artifact_evidence
from metalmind.artifacts.store import atomic_json
from metalmind.benchmarking.baselines import collect_baselines
from metalmind.benchmarking.statistics import summarize_pairs
from metalmind.benchmarking.timer import BenchmarkUnstable, measure_pair
from metalmind.runtime.abi import matches_signature
from metalmind.runtime.environment import probe_environment
from metalmind.runtime.execution import build_execution_callable
from metalmind.runtime.lock import gpu_lock
from metalmind.runtime.metal import MetalBackend
from metalmind.verification.numerics import compare_outputs, inputs_in_domain
from metalmind.verification.policy import policy_for
from metalmind.workloads.fixtures import make_public_fixtures
from metalmind.workloads.registry import resolve_reference

GUARD_SOURCE = """
#include <metal_stdlib>
using namespace metal;

inline bool invalid_fp32(uint bits, uint limit) {
    uint magnitude = bits & 0x7fffffffu;
    return magnitude > limit || (magnitude != 0u && magnitude < 0x00800000u);
}

kernel void check_domain(
    device const uint* x, device const uint* parameter, device atomic_uint* invalid,
    constant uint& elements, constant uint& parameters,
    constant uint& max_x, constant uint& max_parameter,
    uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint grid [[threads_per_grid]]) {
    bool bad = false;
    for (uint i = tid; i < elements; i += grid)
        bad |= invalid_fp32(x[i], max_x);
    for (uint i = tid; i < parameters; i += grid)
        bad |= invalid_fp32(parameter[i], max_parameter);
    if (simd_any(bad) && lane == 0)
        atomic_store_explicit(invalid, 1u, memory_order_relaxed);
}
"""

def bit_pattern(value):
    return int.from_bytes(struct.pack("<f", value), "little")


def make_check(workload):
    policy = policy_for(workload)
    elements = prod(workload.inputs[0].shape)
    parameters = prod(workload.inputs[1].shape) if len(workload.inputs) == 2 else 0
    max_x = bit_pattern(policy.max_input)
    max_parameter = bit_pattern(policy.max_parameter or 0)
    threads = min(65536, ((elements + 1023) // 1024) * 128)
    kernel = MetalBackend().compile(GUARD_SOURCE, entry_point="check_domain")

    def check(inputs):
        flag = torch.zeros(1, dtype=torch.int32, device="mps")
        parameter = inputs[1] if parameters else inputs[0]
        kernel.function(inputs[0], parameter, flag, elements, parameters, max_x, max_parameter,
                        threads=(threads, 1, 1), group_size=(128, 1, 1))
        return flag.item() == 0

    return check


def parity(workload, check, original_inputs):
    policy = policy_for(workload)
    points = 0
    for fixture in make_public_fixtures(workload):
        inputs = tuple(t.to("mps") for t in fixture.inputs)
        assert check(inputs) and inputs_in_domain(workload, inputs)
        points += 1
    for index, t in enumerate(original_inputs):
        limit = policy.max_input if index == 0 else policy.max_parameter
        patterns = (0, 0x80000000, 0x00800000, 0x80800000, bit_pattern(limit),
                    bit_pattern(-limit), bit_pattern(limit)+1, bit_pattern(-limit)+1,
                    1, 0x80000001, 0x7f800000, 0xff800000, 0x7fc00001, 0xffc00001)
        for bits in patterns:
            signed = bits if bits < 1 << 31 else bits - (1 << 32)
            for position in {0, t.numel()//2, t.numel()-1}:
                mutated = t.cpu().clone()
                mutated.view(torch.int32).flatten()[position] = signed
                args = (*original_inputs[:index], mutated.to("mps"), *original_inputs[index+1:])
                assert check(args) == inputs_in_domain(workload, args), (index, bits, position)
                points += 1
    return points


def pair(baseline, candidate, seed):
    try:
        series = measure_pair(baseline, candidate, pairs=10, warmups=100, target_ms=45, seed=seed)
    except BenchmarkUnstable as exc:
        return {"status":"inconclusive", "reason":str(exc)}
    stats = summarize_pairs(series.baseline_ms, series.candidate_ms, seed=seed)
    return {"status":"valid", "baseline_ms":statistics.median(series.baseline_ms),
            "candidate_ms":statistics.median(series.candidate_ms), "summary":stats.model_dump(mode="json"),
            "seed":seed, "series":series.model_dump(mode="json")}


def probe(artifact_path):
    artifact = verify_artifact_evidence(artifact_path)
    workload = artifact.workload
    reference = resolve_reference(workload.reference_id)
    calls = 0

    def fallback(*inputs):
        nonlocal calls
        calls += 1
        return reference(*inputs)

    with gpu_lock(timeout=70), torch.inference_mode():
        check = make_check(workload)
        gpu = MetalBackend().compile(artifact.candidate.proposal.source, entry_point="metalmind_kernel")
        standard = build_execution_callable(workload, gpu, fallback=fallback, domain_mode="checked")
        assumed = build_execution_callable(workload, gpu, fallback=fallback, domain_mode="assume")
        standard.proposal = assumed.proposal = artifact.candidate.proposal
        fixtures = [tuple(t.to("mps") for t in f.inputs) for f in make_public_fixtures(workload)]
        inputs = fixtures[0]
        points = parity(workload, check, inputs)

        def fused(*args):
            if not matches_signature(workload, args) or not check(args):
                return fallback(*args)
            return assumed(*args)

        for fixture in fixtures:
            assert compare_outputs(fused(*fixture), reference(*fixture), policy_for(workload)).passed
        assert calls == 0
        invalid = inputs[0].clone()
        invalid.view(torch.int32).flatten()[0] = 1
        bad = (invalid, *inputs[1:])
        torch.testing.assert_close(fused(*bad), reference(*bad), rtol=0, atol=0)
        assert calls == 1
        baselines, unavailable = collect_baselines(workload)
        old_vs_new = pair(lambda: standard(*inputs), lambda: fused(*inputs), 8430)
        timings = {name:pair(lambda: fn(*inputs), lambda: fused(*inputs), 8431)
                   for name,fn in baselines}
    return {"workload":workload.workload_id,"artifact_id":artifact.artifact_id,"parity_cases":points,
            "fallback_confirmed":True, "old_vs_fused":old_vs_new, "framework":timings,
            "unavailable_baselines":unavailable}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("artifacts", nargs="+", type=Path)
    args = parser.parse_args()
    rows = []
    for path in args.artifacts:
        row = probe(path)
        rows.append(row)
        atomic_json(args.output, {"model_requests":0, "hardware":_hardware_profile(),
                                  "environment":asdict(probe_environment()), "rows":rows})
        def show(data):
            return {k:data[k] for k in ("status","baseline_ms","candidate_ms","summary","reason") if k in data}
        print(row["workload"], show(row["old_vs_fused"]),
              {name:show(data) for name,data in row["framework"].items()}, flush=True)

if __name__=="__main__":
    main()
