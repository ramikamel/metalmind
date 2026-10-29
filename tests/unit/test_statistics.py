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
