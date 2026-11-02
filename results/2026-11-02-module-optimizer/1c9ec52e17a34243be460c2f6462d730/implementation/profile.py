"""Standalone region timings are observations, never additive attribution."""
import os
import statistics
import torch
from metalmind.graph.capture import capture_module
from metalmind.graph.contracts import ModuleAnalysis, LatencySamples
from metalmind.benchmarking.timer import _block, _calibrate, BenchmarkUnstable
from metalmind.runtime.lock import gpu_lock
from metalmind.runtime.worker import gpu_faulted, _record_fault


def assert_measurement_ready():
    if gpu_faulted(): raise ValueError("GPU fault latch is set; run doctor --smoke")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1" or os.environ.get("MTL_SHADER_VALIDATION") not in (None,"0"):
        raise ValueError("Disable CPU fallback and Shader Validation for performance measurement")


def measure_latency(fn, *, samples=10):
    with torch.inference_mode():
        for _ in range(25): fn()
        n = _calibrate(fn,100)
        pilot = [_block(fn,n) for _ in range(5)]
        raw = tuple(_block(fn,n) for _ in range(samples))
        if abs(statistics.median(raw)/statistics.median(pilot)-1) > .1:
            raise BenchmarkUnstable("Standalone measurement drift exceeded 10%; window rejected")
    return LatencySamples(median_ms=statistics.median(raw),samples_ms=raw,repetitions=n)


def analysis_from_capture(captured, *, measure=False):
    whole = None;exclusions = list(captured.exclusions);regions = [r.info for r in captured.regions]
    if measure:
        assert_measurement_ready()
        if any(s.device != "mps" for s in captured.guard.input_specs): raise ValueError("Timing requires MPS inputs")
        with gpu_lock(), torch.inference_mode():
            try:
                whole = measure_latency(lambda:captured.original(*captured.example_inputs))
                for i,region in enumerate(captured.regions):
                    try: regions[i] = region.info.model_copy(update={"latency":measure_latency(lambda:region.function(*region.example_inputs))})
                    except BenchmarkUnstable as exc: exclusions.append(region.info.output_name + ": " + str(exc))
            except BenchmarkUnstable as exc: exclusions.append(str(exc))
            except RuntimeError as exc:
                _record_fault(str(exc));raise
    regions.sort(key=lambda r:(r.latency is not None,r.latency.median_ms if r.latency else r.estimated_input_bytes),reverse=True)
    return ModuleAnalysis(supported=True,graph_hash=captured.graph_hash,inputs=captured.guard.input_specs,
        regions=tuple(regions),exclusions=tuple(exclusions),whole_latency=whole,
        ranking_basis="standalone region latency; not exclusive or additive" if any(r.latency for r in regions) else "estimated input bytes; no timing attribution")


def analyze_module(module, example_inputs, *, measure=False):
    try: return analysis_from_capture(capture_module(module,example_inputs),measure=measure)
    except (ValueError,TypeError,RuntimeError,OSError,AttributeError,torch.fx.proxy.TraceError) as exc:
        return ModuleAnalysis(supported=False,reason=str(exc)[:1000])
