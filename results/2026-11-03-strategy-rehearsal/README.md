# Checked-mode four-strategy rehearsal — November 3, 2026

All four jobs completed and returned **no win**. All seven candidates passed
public numerical verification and obtained valid timing windows. This
development rehearsal used small FP32 bias/GELU, shape (64, 257), on the
14-core M3 Pro. It exercised strategy routing before the frozen 108-job
comparison. Each strategy ran once, so the results describe this signature
and cannot establish a feedback advantage.

| Strategy | Valid candidates | Meta requests | Best candidate vs eager |
| --- | ---: | ---: | ---: |
| Feedback | 2 | 2 | 0.0278× |
| Independent | 2 | 2 | 0.0284× |
| One-shot | 1 | 1 | 0.0286× |
| Template | 2 | 0 | 0.0298× |

The measured boundary includes checked input-domain validation, dispatch,
output allocation and synchronization. Across candidates, ratios ranged
from 0.0278–0.0298× versus eager and 0.0639–0.0686× versus Inductor fullgraph.
All ratios use baseline time divided by candidate time. The search scores
never reached the promotion threshold, so these jobs did not proceed to
private verification or independent confirmation.

The five Muse Wonder requests reported 3,285 input and 8,488 output tokens.
Each request was capped at 4,096 output tokens. The old raw campaign summary
counts local template proposal slots in its requests column. The derived
analysis separates those slots from Meta traffic.

The rehearsal began at source commit `81a735f`. The [manifest](campaign.json)
captures the settings and workload, the [analysis](analysis.json) indexes
all seven measured candidates, and the [hashes](sha256.json) cover raw and
derived data. The [frozen protocol](../../docs/benchmarking.md) is separate.
