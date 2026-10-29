"""Paired ratios and seeded percentile bootstrap; raw samples remain stored."""
import numpy as np
from metalmind.contracts import ConfirmationResult

def summarize_pairs(baseline,candidate,*,seed=42,resamples=10000):
    a=np.asarray(baseline,dtype=float);b=np.asarray(candidate,dtype=float)
    if a.ndim!=1 or a.shape!=b.shape or len(a)<2 or not np.isfinite(a).all() or not np.isfinite(b).all() or (a<=0).any() or (b<=0).any():
        raise ValueError("paired timings must be finite positive vectors of equal length")
    ratios=a/b;rng=np.random.default_rng(seed)
    medians=np.median(ratios[rng.integers(0,len(a),size=(resamples,len(a)))],axis=1)
    lower,upper=np.percentile(medians,[2.5,97.5])
    return ConfirmationResult(speedup=float(np.median(ratios)),lower=float(lower),upper=float(upper),pairs=len(a))
