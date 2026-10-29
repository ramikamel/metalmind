from metalmind.benchmarking.statistics import summarize_pairs


def test_stable_paired_speedup_and_uncertainty():
    result=summarize_pairs([1.1]*30,[1.0]*30,seed=7)
    assert abs(result.speedup-1.1)<1e-10
    assert result.lower>1.0
    tie=summarize_pairs([1.0]*30,[1.0]*30,seed=7)
    assert tie.speedup==1.0 and tie.lower==1.0
