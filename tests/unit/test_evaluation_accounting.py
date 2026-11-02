from metalmind.evaluation import summarize_campaign


def test_failed_jobs_and_fallbacks_stay_in_denominator():
    summary=summarize_campaign([
        {"outcome":"promoted","speedup":1.2},
        {"outcome":"no_win","speedup":1.0},
        {"outcome":"failed","speedup":1.0}])
    assert summary['attempted']==3 and summary['promoted']==1 and summary['failed']==1
    assert abs(summary['geometric_mean_speedup']-1.2**(1/3))<1e-10
