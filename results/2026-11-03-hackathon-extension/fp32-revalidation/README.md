# Exact-source FP32 revalidation

Four previously promoted FP32 artifacts were tested without model calls using the
short-block calibration and fresh private fixtures. Each attempt used 100 warmups,
30 pairs per valid session, and the standard promotion gate. We verified the same
required baseline names before accepting a new promotion. A valid measurement
below the gate stops the attempt. An invalid timing window permits one fresh
attempt. The input domain remained assumed in all cases.

The table reports the strongest verified baseline in each valid session.
`—` means that session was never run.

| Signature | Attempt | Outcome | Session 1, ratio (95% interval) | Session 2, ratio (95% interval) |
| --- | --- | --- | --- | --- |
| `softmax:medium` | 1 | Timing inconclusive | 1.161× (1.117–1.214×) | Invalid timing window |
| `softmax:medium` | 2 | Timing inconclusive | Invalid timing window | — |
| `softmax:large` | 1 | Promoted | 1.173× (1.152–1.188×) | 1.181× (1.177–1.195×) |
| `bias_gelu:large` | 1 | Timing inconclusive | Invalid timing window | — |
| `bias_gelu:large` | 2 | Promoted | 1.080× (1.075–1.086×) | 1.084× (1.079–1.093×) |
| `bias_gelu:medium` | 1 | Promoted | 1.589× (1.573–1.611×) | 1.568× (1.541–1.579×) |

[Summary and raw run paths](summary.json) · [Procedure](procedure.py).

## Medium softmax stability check

After the two unstable formal attempts, a separate in-process diagnostic kept
30 pairs per window and recorded every precheck and timed block. All four
windows were valid. Against eager PyTorch, the two diagnostic ratios were
1.241× and 1.231×. Against Inductor they were 1.893× and 1.923×. The diagnostic
recorded no invalid window and did not isolate a cause for the earlier drift.
[Diagnostic samples and procedure](drift-diagnostic/).

A third formal attempt on November 4 kept the standard drift rule and used
200 warmups, fresh private fixtures and a fresh timing seed. Medium softmax
passed 22 private cases and both sessions. Against eager PyTorch, the fastest
verified baseline, it measured **1.216× (1.158–1.250×) and 1.228× (1.205–1.240×)**. Inductor was
also verified and measured. All four previously promoted FP32 signatures now
have a fresh promotion under the corrected timer. [Third formal attempt](medium-final/).
