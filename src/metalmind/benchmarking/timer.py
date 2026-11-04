"""Synchronized function-level timing; compile/setup are outside warm samples."""
import random
import statistics
import time
import torch
from metalmind.contracts import TimingSeries,TimingReport,BaselineTiming,EvaluationResult
from metalmind.benchmarking.statistics import summarize_pairs

class BenchmarkUnstable(RuntimeError): pass

def _block(fn,repetitions):
    torch.mps.synchronize();start=time.perf_counter()
    for _ in range(repetitions): result=fn()
    torch.mps.synchronize()
    return (time.perf_counter()-start)*1000/repetitions

def _calibrate(fn,target_ms):
    n=1
    while _block(fn,n)*n<target_ms and n<10000: n=min(10000,n*2)
    return n

def measure_pair(baseline,candidate,*,pairs=10,target_ms=100,seed=42,warmups=25):
    with torch.inference_mode():
        for _ in range(warmups): baseline();candidate()
        # Let the slower arm set the common block size to bound asymmetric windows.
        n=min(_calibrate(baseline,target_ms),_calibrate(candidate,target_ms))
        max_attempts=1+(200-warmups)//25
        for attempt in range(max_attempts):
            pilot=[_block(baseline,n) for _ in range(5)]
            if abs(statistics.median(pilot[:2])/statistics.median(pilot[-2:])-1)<=.1: break
            if attempt==max_attempts-1: raise BenchmarkUnstable("baseline warmup remained unstable")
            for _ in range(25): baseline();candidate()
        control=statistics.median(pilot)
        a=[];b=[];order=[];rng=random.Random(seed)
        balanced=[i%2 for i in range(pairs)];rng.shuffle(balanced)
        for first in balanced:
            if first: b.append(_block(candidate,n));a.append(_block(baseline,n));order.append("candidate_first")
            else: a.append(_block(baseline,n));b.append(_block(candidate,n));order.append("baseline_first")
        if abs(statistics.median(a)/control-1)>.1: raise BenchmarkUnstable("baseline drift exceeded 10%; entire window invalid")
    return TimingSeries(baseline_ms=tuple(a),candidate_ms=tuple(b),repetitions=n,order=tuple(order))

def evaluate_performance(request,backend,kernel,verification):
    from metalmind.benchmarking.baselines import collect_baselines
    from metalmind.runtime.execution import build_execution_callable
    from metalmind.workloads.registry import resolve_reference
    from metalmind.workloads.fixtures import make_public_fixtures
    workload=request.workload
    baselines,unavailable=collect_baselines(workload)
    candidate=build_execution_callable(workload,kernel,fallback=resolve_reference(workload.reference_id),domain_mode=request.domain_mode)
    candidate.proposal=request.candidate.proposal
    inputs=tuple(t.to("mps") for t in make_public_fixtures(workload)[0].inputs)
    comparisons=[]
    for name,fn in baselines:
        series=measure_pair(lambda:fn(*inputs),lambda:candidate(*inputs),
            pairs=request.confirmation_pairs if request.mode=="confirm" else request.search_pairs,seed=request.seed,warmups=request.warmups)
        comparisons.append(BaselineTiming(name=name,series=series,
            summary=summarize_pairs(series.baseline_ms,series.candidate_ms,seed=request.seed,resamples=request.bootstrap_resamples)))
    timing=TimingReport(seed=request.seed,bootstrap_resamples=request.bootstrap_resamples,comparisons=tuple(comparisons),unavailable=unavailable,domain_mode=request.domain_mode)
    return EvaluationResult(status="valid",verification=verification,timing=timing,
        elapsed_ms=comparisons[0].series.candidate_ms)
