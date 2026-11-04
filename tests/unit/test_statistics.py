import pytest

from metalmind.benchmarking import timer
from metalmind.benchmarking.statistics import summarize_pairs


def test_stable_paired_speedup_and_uncertainty():
    result=summarize_pairs([1.1]*30,[1.0]*30,seed=7)
    assert abs(result.speedup-1.1)<1e-10
    assert result.lower>1.0
    tie=summarize_pairs([1.0]*30,[1.0]*30,seed=7)
    assert tie.speedup==1.0 and tie.lower==1.0


def test_timing_block_brackets_every_invocation_with_synchronization(monkeypatch):
    from metalmind.benchmarking.timer import _block
    import torch
    events=[]
    monkeypatch.setattr(torch.mps,"synchronize",lambda:events.append("sync"))
    assert _block(lambda:events.append("call"),3)>0
    assert events==["sync","call","call","call","sync"]


@pytest.mark.parametrize(("baseline_ms", "candidate_ms"), [(20., 1.), (1., 20.)])
def test_common_repetitions_target_the_slower_arm(monkeypatch, baseline_ms, candidate_ms):
    def baseline():
        return baseline_ms

    def candidate():
        return candidate_ms

    monkeypatch.setattr(timer, "_block", lambda fn, repetitions: fn())

    series = timer.measure_pair(baseline, candidate, pairs=4, target_ms=100)

    assert series.repetitions == 8
    assert series.baseline_ms == (baseline_ms,) * 4
    assert series.candidate_ms == (candidate_ms,) * 4
    assert series.order.count("baseline_first") == series.order.count("candidate_first") == 2


def test_paired_measurement_rejects_baseline_drift(monkeypatch):
    def baseline():
        pass

    def candidate():
        pass

    baseline_samples = iter([1.] * 5 + [1.2] * 4)
    monkeypatch.setattr(timer, "_calibrate", lambda fn, target_ms: 8)
    monkeypatch.setattr(
        timer, "_block", lambda fn, repetitions: next(baseline_samples) if fn is baseline else 1.
    )

    with pytest.raises(timer.BenchmarkUnstable, match="baseline drift"):
        timer.measure_pair(baseline, candidate, pairs=4)
