# Completed checked-mode FP32 comparison

MetalMind recorded all **108 planned jobs** on a 14-core Apple M3 Pro.
**Zero kernels were promoted.** The four strategies produced 91 no-win
jobs, four budget-exhausted jobs and 13 failed jobs. Each job retained the
original PyTorch path. These results use checked mode, which includes the
input-domain scan in public-call latency.

| Strategy | Jobs | No win | Budget exhausted | Failed | Valid timings | Meta requests |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Feedback | 27 | 23 | 1 | 3 | 40 | 53 |
| Independent | 27 | 24 | 0 | 3 | 44 | 53 |
| One-shot | 27 | 24 | 2 | 1 | 24 | 27 |
| Template | 27 | 20 | 1 | 6 | 38 | 0 |
| **Total** | **108** | **91** | **4** | **13** | **146** | **133** |

Search started 171 candidate evaluations. One interrupted attempt had no
terminal evaluation event. Among the recorded outcomes, seven candidates
failed compilation, eight had runtime errors, two failed correctness, seven
timed out, and 146 passed correctness with valid timings. The minimum ratio
over each candidate's verified baselines ranged from 0.0113× to 0.0915×
across those 146 timings. Ratios divide baseline latency by candidate latency.
The all-job final geometric mean was 1.00× for every strategy because each
job fell back. Winner-only statistics are unavailable.

## Feedback versus independent sampling

All 27 trials could be paired on family, case and trial index. Neither
strategy promoted a candidate in any pair. Of the 22 pairs with comparable
search-stage timings, feedback had the higher best score in 12 and
independent sampling in 10. The other five lacked verified timing on at
least one side. These descriptive search scores stayed below parity and
give no evidence that feedback improved promotion under the tested budget.

## Requests and failures

The ledger recorded 133 Meta requests or reservations against a ceiling of
135. Responses reported 82,462 input and 249,871 output tokens. Those
reported tokens would cost about **$5.33** at the November 3, 2026 standard
Muse Wonder list rates of $4 per million uncached input tokens and $20 per
million output tokens. This estimate treats all reported input as uncached
and excludes requests without reported usage, credits and any other account
adjustments. [Meta pricing](https://dev.meta.ai/models/muse-wonder)

The 13 failed jobs include six worker deadline timeouts, four Meta HTTP 503
errors, one Meta connection failure or timeout, one interruption, and one
local GPU-lock timeout during an overlapping retry. The evaluator serialized
GPU work with its lock, and the ledger retained the affected jobs. Separate
diagnostic reruns used their own ledgers.

The [assumed-domain FP32 pilot](../2026-11-02-fp32-pilot/README.md) used a
different validation boundary and did promote two bias/GELU signatures.

## Reproduce the analysis

The [archive record](archive.json) binds the source commit, package hash,
and request ceiling. The [frozen manifest](campaign.json), [job ledger](summary.json),
[analysis](analysis.json), [matched comparison](comparison.json), and
[499 file checksums](sha256.json) accompany the original raw records.

```sh
campaign=results/2026-11-03-checked-fp32-full
python -m scripts.report_campaign "$campaign" --run-root "$campaign/raw"
python -m scripts.report_campaign_comparison "$campaign" --run-root "$campaign/raw"
```
