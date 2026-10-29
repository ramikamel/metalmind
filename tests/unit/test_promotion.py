from metalmind.benchmarking.promotion import decide_promotion
from metalmind.benchmarking.statistics import summarize_pairs


def test_two_sessions_and_correctness_are_mandatory():
    good=summarize_pairs([1.1]*30,[1.0]*30)
    weak=summarize_pairs([1.04]*30,[1.0]*30)
    assert decide_promotion(True,[[good],[good]]).promote
    assert not decide_promotion(True,[[good]]).promote
    assert not decide_promotion(False,[[good],[good]]).promote
    assert not decide_promotion(True,[[good],[weak]]).promote
