# Implementation progress

Updated November 2, 2026. Plan: [ordered implementation tasks](superpowers/plans/2026-10-29-metalmind.md).

## Current state

The initial function-level optimizer is implemented on `feat/optimizer-core`.
Tasks 1–12 have working code and targeted CPU/GPU verification. Tasks 13–14
have the campaign runner, CLI, reports, and examples; their end-to-end acceptance
is being exercised. Task 15 release verification is in progress. The full
108-job comparison campaign and follow-on graph Tasks 16–17 have not run.
Do not equate implemented infrastructure with all research acceptance criteria.

The authorized nine-case FP32 development pilot is complete: **2 promoted,
7 no-win, 0 failed jobs**, using 18 additional generation requests. Including
seven prior requests, generation usage is **25 of 30 approved requests**; every
request was capped at 4,096 output tokens. No more live requests are required
for this release checkpoint. The access probe is separate.

## Implemented and exercised

- Immutable workload, candidate, evaluation, configuration, and artifact contracts;
  exact shape/stride/dtype/ABI guards and runtime tensor parameters.
- Bias plus tanh GELU, RMSNorm, stable row softmax; registered FP16 variants
  explicitly compute intermediates in FP32. Custom function capture is FP32 only.
- Restricted trusted importable function capture. Static admission precedes FX
  tracing, rejecting side effects, mutation, unsupported calls, and device transfers.
- Spawned GPU workers, exclusive GPU lock, absolute deadlines, memory preflight,
  input integrity, unwritten output/tail checks, and public/private numerical gates.
- Meta Responses API generation, bounded retries, durable pre-dispatch request/token
  reservations, deduplication, feedback, cancellation boundaries, and guarded resume.
- Verified eager, equivalent functional, and compiled baselines; synchronized paired
  timing, raw samples, bootstrap intervals, and two-session promotion.
- Content-bound artifact export and offline loading; mismatches use the original
  function. Explicit numerical audit and full performance revalidation are separate.
- Frozen sequential campaign configuration, all-job denominators, escaped HTML reports,
  developer CLI, CPU-only CI configuration, and package builds.

## Measured development results

Four controller rehearsals preceded the nine-case pilot. One large FP32
bias/GELU signature was promoted in assumed-domain mode; checked medium GELU,
small RMSNorm, and small softmax did not win. All outcomes are retained in
[development evidence](../results/2026-10-29-development-rehearsals/README.md).
The earlier tiny generation probe also compiled and passed numerical tests,
but its checked wrapper was slower and it was not a promoted run.

The large GELU artifact was revalidated after strengthening hardware identity
with the M3 Pro GPU's 14-core configuration. Fresh private verification passed
and two sessions observed 3.079×/3.089× versus eager and 1.081×/1.079× versus
Inductor fullgraph. This is one shape `(512, 4096)`, one chip, FP32, explicit
assumed-domain mode; it is not default checked-mode performance or evidence
that feedback outperforms other search strategies. Derived artifact:
`0fadc5d75491f265155619d741bf48cdc72d11904b4b7c965e60dc63e0177364`.

## Verification and review

Final local suite: **88 passed in 8.54 seconds**, including all real GPU tests,
with no skips. Metal shader health passed. Wheel and source distribution built.
The final wheel's isolated install passed, including shader health and a no-key
108-job campaign preview. [Release records](../results/2026-11-02-release-checks/README.md)
retain code/dependency/wheel identities. Remote CPU CI is checked after pushing.

A fresh independent whole-branch review identified artifact identity, function
capture, deadlines, durable billing reservations, device-fault handling, retry
accounting, and raw-evidence validation issues. Each was reproduced with a failing
regression test and fixed. Measurement settings now propagate end to end,
finalization is terminal/idempotent, and malformed artifact evidence falls back.
Additional regressions cover preprocessing admission, pre-trace side effects,
uint32 element overflow, memory caps, and GPU configuration identity.
Revalidation metadata now retains config/environment/workload/source instead of
overwriting its initial manifest (RED→GREEN). The historical revalidation record
remains unchanged; its artifact and commands supply the available context.
After a GPU fault, new model/evaluation work is blocked until a successful
`doctor --smoke` health check. This is fault containment, not hostile-code isolation.

## Engineering rulings

- Use the user-requested checkout directly on a feature branch. This keeps the
  project at the requested path; concurrent edits there require coordination.
- Use the installed Python 3.14.6 and PyTorch 2.15.0 after capability checks.
  Python 3.12 CPU CI is a separate compatibility check, not GPU validation.
- Keep argparse and environment-report dataclasses; use frozen Pydantic optimizer
  contracts. This limits new dependencies but differs from the original stack sketch.
- Define future-only contracts when their owning subsystem exists. Graph interfaces
  remain planned, rather than shipping empty implementations.
- Treat unstable timing windows as rejections; deterministic unit tests verify
  synchronization order separately. A skipped unstable GPU timing test proves no win.
- Raise controller evaluation timeout from 30 to 120 seconds after real timing
  calibration. One absolute worker deadline still covers startup/lock/compile/execute;
  whole-job and finalization budgets remain enforced.
- Include GPU core count and working-set capability in artifact identity. Older
  fingerprints need explicit revalidation; the derived artifact above passed it.
- Retain private seed descriptors after gate consumption. The original winning
  gate did not retain them; this limitation is disclosed. Its fresh audit and the
  derived revalidation do retain descriptors, published only after search closed.
- Public publication and pushes were explicitly authorized by the user. Credentials
  stay in ignored `.env`; curated public evidence contains synthetic workloads only.

## Remaining acceptance work

Acceptance follow-up: campaign identity v2 now binds resolved workload/reference and
numerical-policy contracts (regression RED→GREEN). The already-running pilot keeps
its original configuration-only identity; its per-run manifests retain workloads.
Reports now include private verification and both saved confirmation sessions, with
stage labels separating search results from final promotion (RED→GREEN).

The [complete nine-case pilot](../results/2026-11-02-fp32-pilot/README.md)
retains all outcomes, source, raw timing, private fixtures, and request/token use.
Medium GELU measured 1.547× / 1.553× versus Inductor; large GELU 1.093× / 1.097×.
All seven no-win results remain in the denominator. Offline new-values/shape/
gradient checks passed without a key. A separate unchanged matmul plus medium
activation measured 1.153× (95% interval 1.142–1.160×) in one 30-pair session.
This is descriptive whole-workflow evidence, not two-session model confirmation.

The full campaign remains outside the current request allowance. Event-specific
model eligibility and submission rules remain unverified because the event was
not identified. Follow-on graph profiling/replacement and a recorded submission
video remain future work. No whole-model optimizer or general kernel support
is claimed. Browser visual preview of the HTML report was blocked by the local
file URL policy; automated report-content and escaping checks passed.
