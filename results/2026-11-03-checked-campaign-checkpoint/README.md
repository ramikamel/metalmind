# Frozen checked-mode FP32 comparison — first 12 of 108 jobs

The first 12 jobs all completed and used the original PyTorch fallback. Of
20 evaluated candidates, 19 passed public numerical checks and yielded valid
timings, and one failed Metal compilation. None of the 19 reached parity
against its strongest verified baseline in search. The worst-baseline
ratios per candidate ranged from 0.0131× to 0.0770×. The ratio is baseline
latency divided by checked public-wrapper latency.

| Strategy | Jobs | Valid timings | Compile errors | Promoted | Meta requests |
| --- | ---: | ---: | ---: | ---: | ---: |
| Feedback | 3 | 6 | 0 | 0 | 6 |
| Independent | 3 | 5 | 1 | 0 | 6 |
| One-shot | 4 | 4 | 0 | 0 | 4 |
| Template | 2 | 4 | 0 | 0 | 0 |
| Total | 12 | 19 | 1 | 0 | 16 |

Muse Wonder reported 10,620 input tokens and 29,728 output tokens. No job
reached private confirmation. All measurements are on the 14-core Apple M3
Pro and include the checked input-domain scan. This partial shuffled prefix
contains no matched feedback/independent trials. Strategy counts are
unequal at this checkpoint. The separate assumed-domain pilot used
a different input-domain validation policy.

[Campaign manifest](campaign.json) freezes the full order, workload and
policy, package hash, environment and request ceiling. The code used for
the run was committed at `c4f3705`. [Summary](summary.json) records all 12
jobs. [Analysis](analysis.json) gives the per-job and strategy metrics, with
raw source and timing events under `raw/`. [Checksums](sha256.json) bind
the 59 archived data files.

Regenerate the analysis without Metal access or a model key:

```sh
checkpoint=results/2026-11-03-checked-campaign-checkpoint
python -m scripts.report_campaign "$checkpoint" --run-root "$checkpoint/raw"
```
