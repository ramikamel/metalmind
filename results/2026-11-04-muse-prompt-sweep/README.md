# Muse prompt sweep across 18 signatures

Muse Wonder ran on all nine FP32 and nine FP16 signatures with a two-request
budget per signature. The revised prompt adds operation-specific Metal hints,
concise baseline feedback, and specific checks for synchronization and compiler
failures from the first prompt study. This is a new search with caller-guaranteed
input bounds. It uses the existing numerical and performance gates.

The sweep recorded **36 Meta requests**, **12 promoted
signatures** and **7 updates to the published best results**.
Promising candidates with invalid confirmation timing received a zero-request
recheck. Existing winners were replaced only when the new artifact cleared
its own promotion gate and beat the earlier artifact directly by at least 2%
in two valid sessions with both lower confidence bounds above 1.00×.
A first promotion for a previously unpromoted signature used the standard gate.

| Signature | Best search ratio | Confirmed ratios, if promoted | Decision |
| --- | ---: | ---: | --- |
| FP16 small bias/GELU | 1.589× | 1.591× / 1.625× | Kept earlier winner |
| FP16 medium bias/GELU | 1.656× | 1.655× / 1.635× | Kept earlier winner |
| FP16 large bias/GELU | 1.867× | 1.857× / 1.765× | Updated published best |
| FP16 small RMSNorm | 1.237× | 1.249× / 1.245× | Updated published best |
| FP16 medium RMSNorm | 2.176× | 2.168× / 2.181× | Updated published best |
| FP16 large RMSNorm | 1.229× | 1.227× / 1.235× | Kept earlier winner |
| FP16 small softmax | 1.167× | 1.151× / 1.164× | Updated published best |
| FP16 medium softmax | 2.636× | 2.657× / 2.703× | Kept earlier winner |
| FP16 large softmax | 2.830× | 2.834× / 2.808× | Updated published best |
| FP32 small bias/GELU | 0.917× | — | Kept previous result |
| FP32 medium bias/GELU | 1.882× | 2.006× / 1.887× | Updated published best |
| FP32 large bias/GELU | 1.081× | 1.078× / 1.080× | Kept earlier winner |
| FP32 small RMSNorm | 0.480× | — | Kept previous result |
| FP32 medium RMSNorm | 0.655× | — | Kept previous result |
| FP32 large RMSNorm | 0.946× | — | Kept previous result |
| FP32 small softmax | 0.527× | — | Kept previous result |
| FP32 medium softmax | 1.219× | — | Timing inconclusive |
| FP32 large softmax | 1.270× | 1.282× / 1.277× | Updated published best |

Search ratios are against each candidate's fastest verified PyTorch baseline.
Confirmed ratios use the strongest verified baseline in each confirmation
session. [Head-to-head data](comparison.json) compares only the two saved
MetalMind artifacts on the same fixtures. These are targeted reruns with a
short budget, so the results do not establish a general prompt advantage.

[Protocol](protocol.json) · [Sweep evidence](summary.json) ·
[Recheck evidence](rechecks/summary.json) · [Procedure](procedure.py)
