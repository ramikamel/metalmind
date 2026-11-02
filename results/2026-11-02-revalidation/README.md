# Hardware-specific revalidation — November 2, 2026

This derives a new artifact from the prior development GELU winner, without
model generation. The environment fingerprint now distinguishes the **14-core
Apple M3 Pro GPU** from other GPU configurations of the same chip family.
The earlier fingerprint intentionally no longer enables reuse.

FP32, shape `(512, 4096)`, bias `(4096,)`, **assumed-domain mode**. Callers guarantee
finite normal/zero bounded inputs; this is not the public default checked mode.
Fresh private verification passed all 22 fixtures. Consumed final fixture seed
descriptors are included because this revalidation is closed and search did not
adapt to the gate.

| Confirmation | Versus eager | Versus Inductor fullgraph |
| --- | ---: | ---: |
| Session 1 | 3.079× (lower 3.069×) | 1.081× (lower 1.076×) |
| Session 2 | 3.089× (lower 3.078×) | 1.079× (lower 1.074×) |

Bounds shown are paired-bootstrap lower 95% bounds; the saved confirmation files
include upper bounds and all 30 raw pairs per baseline. Timings include the
exported public wrapper, output allocation, dispatch, and completion. Compilation
is excluded. These results apply to one signature and environment, not all
workloads or an entire model. They do not establish a causal benefit from feedback.

```sh
metalmind verify --artifact results/2026-11-02-revalidation/artifact
python scripts/verify_offline_reuse.py results/2026-11-02-revalidation/artifact
python -m examples.integrated_block --artifact results/2026-11-02-revalidation/artifact
```

Reuse on other environments remains fallback-only until explicit numerical and
performance revalidation. The source and evidence are preserved together; local
paths in the run record identify the original checkout.
