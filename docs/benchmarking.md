# Measurement contract

The primary boundary is amortized synchronized public-function latency: guards,
configured value-domain checks, fresh output allocation, Python dispatch, and GPU
execution. Inputs are already resident on MPS. Compilation and model latency are
outside warm timing and must not be advertised as free.

Each candidate passes all public numerical and integrity fixtures before timing.
Framework baselines are also verified: original eager, functional RMSNorm where
applicable, and torch.compile/Inductor fullgraph when usable. Unsupported paths
are recorded, not omitted silently. CPU fallback is disallowed during comparison.

Use at least 25 warmups, calibrated common repetitions, balanced pair ordering,
ten search pairs, and thirty independent confirmation pairs. Bootstrap whole
pairs 10,000 times. Promotion requires at least 1.05x median public-function gain
and lower 95% bound above 1.0 in both confirmation sessions, separated by at least
60 seconds. A private numerical gate precedes confirmation. Drift beyond 10%
invalidates the window; no individual unfavorable samples are removed.

Checked mode scans input domains and includes scan/synchronization cost. Assumed
mode requires caller-guaranteed numerical bounds; its numbers are labeled
separately. The first tiny checked-mode kernel was slower despite being correct;
that run is development data and does not establish a general performance claim.

## Frozen FP32 comparison

The suite in `configs/evaluation.yaml` uses checked mode and freezes 108 jobs
across nine signatures, three trials and four search strategies. Feedback and
independent sampling each get up to two requests and two candidates. One-shot
gets one request and one candidate. Template tuning evaluates its first two
predeclared group sizes, 32 and 64, locally. The Meta request ceiling is
135 (27 × (2 + 2 + 1)), with 4,096 output tokens per request. Retries draw from
the same slots. Each candidate faces the same numerical evaluator and promotion
policy. The earlier assumed-domain pilot is a separate experiment.

```sh
metalmind evaluate --suite configs/evaluation.yaml
metalmind evaluate --suite configs/evaluation.yaml --execute --max-jobs 12
metalmind evaluate --suite configs/evaluation.yaml --execute --resume --max-jobs 12
```

Preview resolves workload shapes, references, and correctness policies without
contacting Meta. Execution records the shuffled job order, package hash and
environment fingerprint under `runs/campaigns/<campaign-id>/`. Resume uses the
same checkout, configuration and Apple Silicon environment. Each new job is
journaled before search. A terminal result can be recovered if the process
stopped before updating the campaign summary. An interrupted nonterminal job
counts as a failed trial with its recorded reservation and is skipped on resume.
Template slots and Meta requests are counted separately. A GPU fault pauses
execution pending a successful `metalmind doctor --smoke`.

The campaign aggregate assigns unsuccessful jobs a neutral factor of 1.0 and
uses each winner's worst confirmed baseline/session ratio. Its geometric mean
describes normalized local results. The integrated workflow is measured
separately by `examples.integrated_block`, using unchanged eager matmul plus
the original activation as its baseline.
