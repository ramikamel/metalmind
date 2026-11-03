import pytest

from scripts.report_workflow_amortization import estimate_amortization


def test_workflow_amortization_uses_paired_savings_and_rounds_calls_up():
    result = estimate_amortization([3.0, 2.0, 1.0], [1.0, 3.0, 0.5], 0.0011)
    assert result["median_paired_saving_ms"] == 0.5
    assert result["calls_to_offset_recorded_cost"] == 3


def test_nonpositive_paired_savings_has_no_break_even():
    result = estimate_amortization([1.0, 2.0], [1.0, 3.0], 0.1)
    assert result["calls_to_offset_recorded_cost"] is None


@pytest.mark.parametrize("baseline,candidate,seconds", [
    ([1.0, 2.0], [1.0], 1.0),
    ([1.0, float("nan")], [1.0, 2.0], 1.0),
    ([1.0, 2.0], [1.0, 2.0], -1.0),
])
def test_invalid_amortization_inputs_are_rejected(baseline, candidate, seconds):
    with pytest.raises(ValueError):
        estimate_amortization(baseline, candidate, seconds)
