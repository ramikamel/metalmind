import pytest
import torch
from metalmind.benchmarking.timer import measure_pair
pytestmark=[pytest.mark.gpu,pytest.mark.skipif(not torch.backends.mps.is_available(),reason="MPS unavailable")]


def test_measurement_waits_for_complete_work():
    x=torch.ones(1024,device="mps")
    series=measure_pair(lambda:x+1,lambda:x+2,pairs=2,target_ms=10)
    assert len(series.baseline_ms)==len(series.candidate_ms)==2
    assert all(v>0 for v in series.baseline_ms+series.candidate_ms)
    assert series.repetitions>=1
