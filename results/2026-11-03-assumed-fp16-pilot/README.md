# Completed assumed-domain FP16 pilot

MetalMind promoted **6 of 9 signatures** across bias/GELU, RMSNorm and
softmax on a 14-core Apple M3 Pro. The kernels use FP32 intermediates and
return FP16. The measured calls assume the caller guarantees the registered
input bounds. The checked-mode FP32 campaign used a different validation
boundary.

| FP16 signature | Shape | Outcome | Minimum speedup across verified baselines, sessions 1 / 2 |
| --- | --- | --- | --- |
| Bias/GELU small | (64, 257) | Inconclusive | — |
| Bias/GELU medium | (256, 1024) | Promoted | 1.657× / 1.651× |
| Bias/GELU large | (512, 4096) | Promoted | 1.240× / 1.242× |
| RMSNorm small | (64, 257) | Inconclusive | — |
| RMSNorm medium | (256, 1024) | Promoted | 1.541× / 1.530× |
| RMSNorm large | (512, 4096) | Inconclusive | — |
| Softmax small | (64, 257) | Promoted | 1.113× / 1.119× |
| Softmax medium | (256, 1024) | Promoted | 2.641× / 2.544× |
| Softmax large | (128, 4096) | Promoted | 1.428× / 1.420× |

Eager PyTorch was the strongest verified baseline for the small and medium
softmax winners. Inductor fullgraph was strongest for the other four. Medium
softmax also measured 2.775× and 2.638× against Inductor in the two sessions.
Its lower 95% interval bounds against eager were 2.622× and 2.511×.
Each promotion passed 22 private cases and two 30-pair confirmation sessions.

There was one feedback trial per signature with two Meta request slots.
All 18 requests returned responses, one proposal failed schema admission,
and all 17 evaluated candidates passed public correctness and yielded valid
search timings. Three confirmation outcomes were inconclusive. Small
bias/GELU and small RMSNorm lost a timing window to baseline drift. Large
RMSNorm passed private verification, then reached the confirmation worker
deadline. The Metal health check passed before the final job resumed.

A [separate zero-request recheck](../2026-11-03-fp16-confirmation-recheck/README.md)
later promoted two of the three saved candidates. The original pilot table
preserves the outcomes measured under its original timing protocol.

The all-job geometric mean of the worst confirmed ratios was **1.325×**,
with a factor of 1.0 assigned to each fallback. This is a normalized summary
of the nine jobs. The design uses one strategy and one trial per signature.

## Offline artifact reuse

All six promoted artifacts were loaded without a model key and tested on
new in-domain values. Each audit observed exactly one Metal dispatch for
the matching FP16 call and zero further dispatches for changed shape, FP32
inputs and gradient mode. The greatest absolute error across the six audits
was 2.3842e-7. [Reuse audit](reuse-audit.json)

## Evidence

[Manifest](campaign.json) · [Job ledger](summary.json) · [Analysis](analysis.json)
· [Comparison](comparison.json) · [Raw evidence checksums](sha256.json)

Muse Wonder reported 18,973 input and 40,555 output tokens. The reported tokens
would cost about $0.89 at the November 3, 2026 standard list rates assuming
uncached input, before credits or other account adjustments. [Meta pricing](https://dev.meta.ai/models/muse-wonder)

```sh
campaign=results/2026-11-03-assumed-fp16-pilot
python -m scripts.report_campaign "$campaign" --run-root "$campaign/raw"
python -m scripts.audit_fp16_reuse "$campaign/raw/runs/fp16-pilot/artifacts/bc69c1dd244e635636f04c7f16e7eb4a9aaec6080fad14af949ff399dfc352e2"
```
