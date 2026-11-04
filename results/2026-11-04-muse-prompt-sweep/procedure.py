"""Run the revised Muse prompt on all registered FP16 and FP32 signatures."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from metalmind.agent import meta
from metalmind.agent.controller import SearchController
from metalmind.agent.prompts import build_prompt
from metalmind.artifacts.export import _hardware_profile, finalize_run, verify_artifact_evidence
from metalmind.artifacts.store import ArtifactStore, atomic_json
from metalmind.config import RunConfig
from metalmind.demonstration.provenance import runtime_source_info
from metalmind.runtime.worker import gpu_faulted
from metalmind.workloads.registry import get_workload

TARGETS = (
    ('rmsnorm', 'small', 'float32', ('tuned',)),
    ('softmax', 'small', 'float32', ('tuned',)),
    ('bias_gelu', 'small', 'float32', ('tuned',)),
    ('rmsnorm', 'medium', 'float32', ('tuned',)),
    ('rmsnorm', 'large', 'float32', ('tuned',)),
    ('softmax', 'small', 'float16', ('tuned',)),
    ('rmsnorm', 'small', 'float16', ('tuned',)),
    ('bias_gelu', 'large', 'float32', ('tuned',)),
    ('softmax', 'large', 'float32', ('tuned',)),
    ('softmax', 'medium', 'float32', ('tuned',)),
    ('bias_gelu', 'large', 'float16', ('tuned',)),
    ('rmsnorm', 'large', 'float16', ('tuned',)),
    ('softmax', 'large', 'float16', ('tuned',)),
    ('rmsnorm', 'medium', 'float16', ('tuned',)),
    ('bias_gelu', 'medium', 'float32', ('tuned',)),
    ('bias_gelu', 'medium', 'float16', ('tuned',)),
    ('bias_gelu', 'small', 'float16', ('tuned',)),
    ('softmax', 'medium', 'float16', ('tuned',)),
)
SETTINGS = dict(
    model="muse-wonder-1.0", reasoning_effort="medium", domain_mode="assume",
    max_requests=2, max_candidates=2, max_output_tokens=8192,
    total_output_tokens=16384, request_timeout_seconds=180, compile_timeout_seconds=90,
    execution_timeout_seconds=300, wall_seconds=1500, finalization_reserve_seconds=600,
    warmups=100,
)
HINTS = {
    "bias_gelu": (
        "Fuse the bias add and GELU into one pass. For narrow or odd rows, compare flat scalar "
        "threads with several adjacent columns per thread. Use vector loads only when every row "
        "address is aligned and handle all tail elements. Minimize indexing and repeated loads."
    ),
    "rmsnorm": (
        "Accumulate squares in FP32. Consider SIMD reductions with a shared-memory merge across "
        "SIMD groups, or independent rows per SIMD group if shape permits. Keep input values or "
        "weight vectors in registers only when the register cost is reasonable. Use one reciprocal "
        "square root per row and avoid redundant reads and threadgroup barriers."
    ),
    "softmax": (
        "Use stable row max and sum reductions in FP32. Consider one SIMD group per row when it "
        "can hold its values efficiently, or a multi-SIMD row reduction when width warrants it. "
        "If values fit in registers, reuse them across both reductions and normalization."
    ),
}


def compact_feedback(feedback):
    compact = []
    for message in feedback:
        head, sep, payload = message.partition("; baseline comparisons ")
        if not sep:
            compact.append(message)
            continue
        timing = json.loads(payload)
        comparisons = timing["comparisons"]
        decisive = min(comparisons, key=lambda row: row["summary"]["speedup"])
        speedups = ", ".join(
            f"{row['name']} {row['summary']['speedup']:.3f}x" for row in comparisons
        )
        compact.append(f"{head}; {speedups}. Strongest baseline: {decisive['name']}.")
    return tuple(compact)


def tuned_prompt(context):
    shortened = context.model_copy(update={"feedback": compact_feedback(context.feedback)})
    base = build_prompt(shortened)
    base = base.replace(
        "Start with a simple correct implementation, conservative threadgroups, bounds checks and uniform barriers.",
        "Choose a launch for this shape with bounds checks and uniform barriers.",
        1,
    )
    goal = (
        "Target: Apple M3 Pro, 14 GPU cores. Optimize synchronized public-call latency against "
        "the fastest verified PyTorch baseline. Inductor and functional RMSNorm can beat eager "
        "PyTorch. This experiment assumes callers guarantee the value domain. "
        "Use the runtime SIMD width when needed. Think about memory passes, synchronization and "
        "work per thread. Pick one concrete hypothesis. If feedback shows a compile or correctness "
        "failure, fix it first. If feedback shows a correct but slow kernel, target the strongest "
        "baseline and change one major aspect of the schedule."
    )
    return (base + "\nOptimization guidance: " + goal + "\nOperation guidance: " + HINTS[context.workload.family]
            + "\nUse consistent scalar or vector grid coordinate types. Use parentheses around a vector pointer cast before indexing. "
              "For shared reductions in multiple phases use distinct scratch or a uniform barrier after all reads before reuse.")


def verify_rendering():
    from metalmind.agent.provider import ProposalContext

    for family, case, dtype, _ in TARGETS:
        context = ProposalContext(workload=get_workload(family, case, dtype))
        current = build_prompt(context)
        tuned = tuned_prompt(context)
        for line in current.splitlines():
            if line.startswith(("ABI:", "Semantics:", "Workload:", "Finite normal", "Return ONLY")):
                assert line in tuned, line
        assert HINTS[family] in tuned and "fastest verified PyTorch baseline" in tuned
    fake = 'Verified candidate 123; baseline comparisons {"comparisons":[{"name":"eager","summary":{"speedup":2.5}},{"name":"inductor","summary":{"speedup":0.9},"series":{"candidate_ms":[100]}}]}'
    assert compact_feedback((fake, "compile_error: diagnostic")) == (
        "Verified candidate 123; eager 2.500x, inductor 0.900x. Strongest baseline: inductor.",
        "compile_error: diagnostic",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path)
    args = parser.parse_args()
    verify_rendering()
    if args.output is None:
        print("Verified rendering and correctness boundaries for both prompts.")
        return
    root = args.output.resolve()
    if (root / "protocol.json").exists():
        parser.error("Output already has a protocol")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    atomic_json(root / "protocol.json", {
        "targets": TARGETS,
        "settings": SETTINGS,
        "max_total_requests": 36,
        "pair_seeds": [202611450 + i * 10 for i in range(len(TARGETS))],
        "hardware": _hardware_profile(),
        "source_commit": source,
        "runtime_source": runtime_source_info(),
        "primary_outcome": "promotions against all verified baselines",
        "secondary_outcomes": ["numerically valid candidates", "valid timed candidates", "best search ratio"],
    })
    rows = []
    for index, (family, case, dtype, arms) in enumerate(TARGETS):
        workload = get_workload(family, case, dtype)
        for arm in arms:
            job_root = root / "runs" / f"{dtype}-{family}-{case}" / arm
            config = RunConfig(**SETTINGS, seed=202611450 + 10 * index,
                               runs_dir=job_root / "jobs", artifacts_dir=job_root / "artifacts")
            prompts = []
            renderer = build_prompt if arm == "current" else tuned_prompt

            def capture(context):
                prompt = renderer(context)
                prompts.append({"sha256": hashlib.sha256(prompt.encode()).hexdigest(), "input": prompt})
                atomic_json(job_root / "prompts.json", prompts)
                return prompt

            print(f"Starting {dtype} {family} {case} with {arm}", flush=True)
            with patch.object(meta, "build_prompt", capture):
                summary = SearchController().run(workload, config)
            result = finalize_run(summary)
            events = ArtifactStore(summary.run_path).read_events()
            evaluations = [event["result"] for event in events if event["event"] == "evaluation"]
            usage = [event["usage"] for event in events if event["event"] == "model_response"]
            timings = []
            for path in sorted(result.run_path.glob("confirmation-*.json")):
                result_data = json.loads(path.read_text())
                comparisons = (result_data.get("timing") or {}).get("comparisons", [])
                timings.append(min(comparisons, key=lambda c: c["summary"]["speedup"]) if comparisons else None)
            if result.artifact_path:
                verify_artifact_evidence(result.artifact_path)
            row = {
                "arm": arm, "family": family, "case": case, "dtype": dtype, "requests": summary.requests,
                "evaluated": summary.evaluated, "numerically_valid": sum(
                    bool((e.get("verification") or {}).get("passed")) for e in evaluations),
                "valid_timed": sum(bool(e.get("timing")) for e in evaluations),
                "evaluation_outcomes": dict(Counter(e["status"] for e in evaluations)),
                "outcome": result.outcome, "reason": result.reason, "search_score": summary.incumbent_score,
                "run_path": str(result.run_path.relative_to(root)),
                "artifact_path": str(result.artifact_path.relative_to(root)) if result.artifact_path else None,
                "confirmation": [{"baseline": c["name"], **c["summary"]} if c else None for c in timings],
                "model_input_tokens": sum(u.get("input_tokens") or 0 for u in usage),
                "model_output_tokens": sum(u.get("output_tokens") or 0 for u in usage),
            }
            rows.append(row)
            atomic_json(root / "summary.json", {"rows": rows, "requests": sum(r["requests"] for r in rows)})
            print(json.dumps(row), flush=True)
            if gpu_faulted():
                subprocess.run([sys.executable, "-m", "metalmind.cli", "doctor", "--smoke", "--json"], check=True)


if __name__ == "__main__":
    main()
