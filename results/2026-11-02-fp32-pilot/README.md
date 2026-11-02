# Nine-case FP32 development pilot — November 2, 2026

**2 of 9 signatures promoted; 7 no-win; 0 failed jobs.** One feedback trial per
signature, at most two model requests/candidates each, `muse-wonder-1.0`, a
14-core Apple M3 Pro GPU, PyTorch 2.15.0, **assumed-domain mode**. The pilot used
18 requests capped at 4,096 output tokens each, bringing session generation
usage to 25 of 30 authorized requests. Reported usage: 18,915 input and 42,146
output tokens. No monetary price is assumed.

This is development evidence, not the full 108-job multi-strategy campaign. It
does not establish that feedback outperforms one-shot or independent sampling.
Checked mode remains the public default; these numbers require caller-guaranteed
finite bounded normal/zero inputs and do not measure checked-mode scans.

| Signature | Input shape | Outcome | Versus Inductor fullgraph, sessions 1 / 2 |
| --- | --- | --- | --- |
| [bias_gelu small](jobs/bias_gelu-small/report.html) | (64, 257) | no_win | — |
| [bias_gelu medium](jobs/bias_gelu-medium/report.html) | (256, 1024) | promoted | 1.547× / 1.553× |
| [bias_gelu large](jobs/bias_gelu-large/report.html) | (512, 4096) | promoted | 1.093× / 1.097× |
| [rmsnorm small](jobs/rmsnorm-small/report.html) | (64, 257) | no_win | — |
| [rmsnorm medium](jobs/rmsnorm-medium/report.html) | (256, 1024) | no_win | — |
| [rmsnorm large](jobs/rmsnorm-large/report.html) | (512, 4096) | no_win | — |
| [softmax small](jobs/softmax-small/report.html) | (64, 257) | no_win | — |
| [softmax medium](jobs/softmax-medium/report.html) | (256, 1024) | no_win | — |
| [softmax large](jobs/softmax-large/report.html) | (128, 4096) | no_win | — |

Both winners passed **22 private fixtures** and **two confirmation sessions**, each
with 30 balanced pairs per baseline and 10,000 paired bootstrap resamples. The
medium GELU lower 95% bounds versus Inductor were 1.534× and 1.543×; large GELU
bounds were 1.086× and 1.089×. Corresponding eager gains were 2.372× / 2.360×
(medium) and 3.051× / 3.070× (large). Compilation is outside warm timing; guards,
allocation, dispatch, and completion are included in the public-call boundary.

All 18 requests are accounted for: 17 candidates were evaluated, with 14 valid
measurements, two entire windows rejected for baseline drift, one nonfinite-output
numerical rejection, and one incomplete proposal rejected before evaluation.
No samples were removed from an accepted window. Final private seed descriptors
are published only because all pilot searches and gates are now closed.

The all-job normalized geometric mean is **1.060×**, assigning each no-win a neutral
1.0; winner-only is 1.300×. These are normalized research summaries, not measured
combined-workload latency. The nine job directories retain manifests, events,
source, final outcomes, raw samples, and available private/confirmation evidence.
`analysis.json` gives all denominators, token use, and exact intervals.

## Offline and integrated checks

The medium artifact was loaded in a fresh process outside the checkout, with
no model key or local `.env`. New values matched (max absolute error 4.768e-7),
and changed shapes and gradients used the original fallback. See
[pilot-offline-reuse.json](pilot-offline-reuse.json).

An unchanged eager matmul plus the guarded medium activation measured **1.153×**
whole-workflow speedup, 95% interval **1.142–1.160×**, in one 30-pair session. Full
outputs matched. This is a separate descriptive experiment, not two-session
whole-model confirmation or a compiled whole-workflow comparison. Raw pairs:
[pilot-integrated-block.json](pilot-integrated-block.json).

```sh
python scripts/verify_offline_reuse.py results/2026-11-02-fp32-pilot/artifacts/db066f6dfe2e74cefc0c335852229dd2ed6a9058518279eabb7b3a28342caa87
python -m examples.integrated_block --artifact results/2026-11-02-fp32-pilot/artifacts/db066f6dfe2e74cefc0c335852229dd2ed6a9058518279eabb7b3a28342caa87
```

These commands make no model requests. Artifact reuse requires the matching
hardware/software fingerprint; another environment requires explicit revalidation.
The saved report links describe the original checkout's paths.

Pilot ID: `4ba23d2ada20e97c974af64e800f924d46679de8d0bfe53935b46f26bd0e16e8`.
It began under configuration-only campaign identity version 1. Per-job manifests
retain the actual workload/reference/policy. Future version-2 campaign IDs also
bind those resolved contracts and therefore differ from this historical ID.
Re-running a campaign incurs new requests and needs a separate output location;
the runner never overwrites an existing campaign directory.
