"""Recheck one saved FP16 kernel with common repetitions capped by the slower arm."""
import hashlib
import json
from pathlib import Path
import random
import secrets
import statistics
import subprocess
import sys
import time

from metalmind.artifacts.export import export_artifact
from metalmind.artifacts.store import atomic_json
from metalmind.benchmarking.promotion import decide_promotion
from metalmind.contracts import CandidateSpec, EvaluationRequest, EvaluationResult, TimingSeries, WorkloadSpec
from metalmind.runtime.lock import gpu_lock
from metalmind.runtime.worker import _child, _record_fault
from metalmind.verification.final_gate import run_final_gate


def read(path):
    return json.loads(path.read_text())


def worker(request_path, result_path):
    import torch
    from metalmind.benchmarking import baselines, timer

    diagnostics = []
    names = iter(())
    original_collect = baselines.collect_baselines

    def collect(*args, **kwargs):
        nonlocal names
        available, unavailable = original_collect(*args, **kwargs)
        names = iter(name for name, _ in available)
        return available, unavailable

    def measure(baseline, candidate, *, pairs=30, target_ms=100, seed=42, warmups=100):
        name = next(names)
        with torch.inference_mode():
            for _ in range(warmups):
                baseline()
                candidate()
            baseline_reps = timer._calibrate(baseline, target_ms)
            candidate_reps = timer._calibrate(candidate, target_ms)
            reps = min(baseline_reps, candidate_reps)
            diagnostic = {
                "baseline": name, "baseline_calibration": baseline_reps,
                "candidate_calibration": candidate_reps, "repetitions": reps,
                "warmup_controls_ms": [],
            }
            diagnostics.append(diagnostic)
            max_attempts = 1 + (200 - warmups) // 25
            for attempt in range(max_attempts):
                controls = [timer._block(baseline, reps) for _ in range(5)]
                diagnostic["warmup_controls_ms"].append(controls)
                if abs(statistics.median(controls[:2]) / statistics.median(controls[-2:]) - 1) <= .1:
                    break
                if attempt == max_attempts - 1:
                    diagnostic["window_valid"] = False
                    raise timer.BenchmarkUnstable(f"{name}: baseline warmup remained unstable")
                for _ in range(25):
                    baseline()
                    candidate()
            order = [i % 2 for i in range(pairs)]
            random.Random(seed).shuffle(order)
            a, b, labels = [], [], []
            for first in order:
                if first:
                    b.append(timer._block(candidate, reps))
                    a.append(timer._block(baseline, reps))
                    labels.append("candidate_first")
                else:
                    a.append(timer._block(baseline, reps))
                    b.append(timer._block(candidate, reps))
                    labels.append("baseline_first")
            drift = statistics.median(a) / statistics.median(controls) - 1
            diagnostic.update({"baseline_ms": a, "candidate_ms": b, "order": labels,
                               "baseline_drift_fraction": drift, "window_valid": abs(drift) <= .1})
            if abs(drift) > .1:
                raise timer.BenchmarkUnstable(f"{name}: baseline drift {drift:+.2%} exceeded 10%")
        return TimingSeries(baseline_ms=a, candidate_ms=b, repetitions=reps, order=labels)

    class Connection:
        def send(self, message):
            if "result" in message:
                atomic_json(result_path, {"result": message["result"], "diagnostics": diagnostics})

        def close(self):
            pass

    baselines.collect_baselines = collect
    timer.measure_pair = measure
    _child(read(request_path), Connection())


def main():
    output = Path(__file__).resolve().parent
    pilot = Path("results/2026-11-03-assumed-fp16-pilot")
    row = next(r for r in read(pilot/"summary.json")["rows"]
               if (r["family"], r["case"]) == ("rmsnorm", "large"))
    archived = pilot/"raw"/row["run_path"]
    manifest = read(archived/"manifest.json")
    workload = WorkloadSpec.model_validate(manifest["workload"])
    candidate = CandidateSpec.model_validate(manifest["archive"][0]["candidate"])
    source = archived/"candidates"/f"{candidate.candidate_id}.metal"
    assert source.read_text() == candidate.proposal.source
    search, = [e for e in map(json.loads, (archived/"events.jsonl").read_text().splitlines())
               if e.get("candidate_id") == candidate.candidate_id and e["event"] == "evaluation"]
    required = {c["name"] for c in search["result"]["timing"]["comparisons"]}
    atomic_json(output/"protocol.json", {
        "source_run": row["run_path"], "candidate_id": candidate.candidate_id,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "domain": "assume", "calibration_policy": "minimum of the two calibrated repetition counts",
        "target_ms_for_slower_arm": 100, "pairs": 30, "warmups": 100, "cooldown_seconds": 60,
        "confirmation_sessions": 2, "retry_invalid_session_at_most": 1,
        "max_baseline_drift": .10, "required_baselines": sorted(required),
        "model_requests": 0,
    })
    gate = run_final_gate(candidate, workload, tuple(secrets.randbits(63) for _ in range(16)),
                          marker_path=output/"private-gate-consumed", timeout=120)
    atomic_json(output/"private-verification.json", gate.model_dump(mode="json"))
    if gate.status != "valid" or gate.verification.cases != 22:
        raise RuntimeError(f"Fresh private correctness gate failed: {gate.reason}")
    confirmations = []
    attempts = []
    for session in (1, 2):
        if session > 1:
            time.sleep(60)
        for attempt in (1, 2):
            if attempt > 1:
                time.sleep(60)
            request = EvaluationRequest(
                candidate=candidate, workload=workload, mode="confirm", measure=True,
                domain_mode="assume", seed=33042 + session * 10 + attempt, warmups=100,
                confirmation_pairs=30, bootstrap_resamples=10000,
            )
            stem = f"session-{session}-attempt-{attempt}"
            request_path, result_path = output/f"{stem}-request.json", output/f"{stem}.json"
            atomic_json(request_path, request.model_dump(mode="json"))
            with gpu_lock(30):
                try:
                    subprocess.run(
                        [sys.executable, __file__, "--worker", str(request_path), str(result_path)],
                        check=True, timeout=300,
                    )
                except subprocess.TimeoutExpired:
                    _record_fault("Instrumented benchmark worker timed out")
                    raise
            report = EvaluationResult.model_validate(read(result_path)["result"])
            attempts.append({"session": session, "attempt": attempt, "record": result_path.name,
                             "status": report.status, "reason": report.reason})
            print(stem, report.status, report.reason, flush=True)
            if report.status == "valid":
                if {c.name for c in report.timing.comparisons} != required:
                    raise ValueError("Confirmation baseline set differs from the original search")
                confirmations.append(report.timing)
                break
            if report.device_fault:
                _record_fault(report.reason)
                raise RuntimeError(report.reason)
            if not (report.reason or "").startswith("BenchmarkUnstable:"):
                raise RuntimeError(report.reason)
        if len(confirmations) != session:
            break
    artifact = None
    outcome = "inconclusive"
    if len(confirmations) == 2:
        decision = decide_promotion(gate.verification, [[c.summary for c in t.comparisons] for t in confirmations])
        outcome = "promoted" if decision.promote else "no_win"
        if decision.promote:
            artifact = export_artifact(candidate, workload, gate.verification, confirmations,
                                       root=output/"artifacts", domain_mode="assume", required_baselines=required)
    atomic_json(output/"summary.json", {
        "outcome": outcome, "attempts": attempts, "artifact_path": str(artifact) if artifact else None,
        "model_requests": 0,
    })
    print(outcome, flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        main()
