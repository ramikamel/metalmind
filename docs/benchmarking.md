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
ten search pairs, and thirty independent confirmation pairs. Use the smaller
of the two calibrated repetition counts so the slower arm sets the block
duration. Earlier frozen runs used the larger count. Bootstrap whole
pairs 10,000 times. Promotion requires at least 1.05x median public-function gain
and lower 95% bound above 1.0 in both confirmation sessions, separated by at least
60 seconds. A private numerical gate precedes confirmation. Drift beyond 10%
invalidates the window; no individual unfavorable samples are removed.

Checked mode scans input domains and includes scan and synchronization costs.
The scan uses IEEE magnitude bits so FP32 subnormals stored on the GPU cannot
pass as zero under MPS floating-point comparisons. Assumed-domain mode requires
caller-guaranteed numerical bounds and its numbers are labeled separately.
The first tiny checked-mode kernel was slower despite passing correctness.
That development run does not establish a general performance claim.

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

The [first 12 jobs](../results/2026-11-03-checked-campaign-checkpoint/README.md)
are archived with their original proposal and timing records. Generate a
read-only analysis of the current ledger with
`python -m scripts.report_campaign runs/campaigns/<campaign-id>`.
The report separates search-stage thresholds from confirmed promotion and
pairs feedback with independent sampling only at matching signature/trial
keys.

## Reuse cost from saved data

The [workflow amortization report](../results/2026-11-03-workflow-amortization/README.md)
uses the original 30 timing pairs and the recorded search plus confirmation
cooldown. The 4.45-million-call estimate covers that partial setup cost at
the observed 29.1 µs median paired saving. Five separate loads of the existing
artifact measured a median 199.63 ms, or about 6,863 workflow calls to offset
that load. Other finalization time remains unavailable in the original run.


## Closeout and separate FP16 pilot

Run `python -m scripts.report_campaign_comparison <campaign-dir> --run-root
<execution-checkout>` to review planned and attempted denominators, promotion
rates, and confirmed all-job and winner-only aggregates. Feedback and
independent sampling pair on family, case and trial. Search timings compare
only when both selected candidates faced the same verified baselines. Rate
intervals use a binomial model and the comparisons are descriptive.

After all jobs finish, use `python -m scripts.archive_campaign <campaign-dir>
--run-root <execution-checkout> --destination <new-results-directory>`. The
archive checks the frozen package fingerprint, reconciles each recorded result
and request count with the copied evidence, and regenerates both reports.
Use relative run and artifact paths in the campaign configuration so the
archive can preserve a portable layout.

`configs/fp16-pilot.yaml` freezes a separate nine-signature, one-trial feedback
pilot with up to 18 Meta requests. It measures FP16 with FP32 intermediates
and caller-guaranteed input bounds. The FP32 checked campaign uses a different
dtype and validation boundary. Keep the results in separate reports.
