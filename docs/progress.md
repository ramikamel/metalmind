# Implementation progress

Updated November 3, 2026. Plan: [ordered implementation tasks](superpowers/plans/2026-10-29-metalmind.md).

## Current state

The function-level release is published on `main` (`cdb8ec1`); its CPU CI passed.
Module continuation is implemented on `feat/module-optimizer` (core `e34a32f`).
Original Tasks 16–17 now have tested code and measured acceptance for flat pure
FP32 inference modules, not arbitrary nested models. Tasks 13–15 still have
broader research/submission items: the full 108-job campaign and authenticated
submission. The later compiled whole-module comparison supersedes the earlier
eager-only acceptance; details are recorded below.

The authorized nine-case FP32 development pilot is complete: **2 promoted,
7 no-win, 0 failed jobs**, using 18 additional generation requests. Including
seven prior requests, generation usage is **25 of 30 approved requests**; every
request was capped at 4,096 output tokens. No more live requests are required
for this release checkpoint. The access probe is separate.

## Implemented and exercised

- Immutable workload, candidate, evaluation, configuration, and artifact contracts;
  exact shape/stride/dtype/ABI guards and runtime tensor parameters.
- Bias plus tanh GELU, RMSNorm, stable row softmax; registered FP16 variants
  explicitly compute intermediates in FP32. Custom FP16 capture preserves that cast order.
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
- Flat pure FP32 module capture, exact region extraction, standalone profiling and
  guarded artifact rewrites with a separate private/whole-module confirmation gate.
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

Function-level checkpoint: **88 passed in 8.54 seconds**, including all real GPU tests,
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
  were added when the module subsystem was implemented, rather than as empty stubs.
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

The full campaign remains outside the current request allowance. The event was
subsequently identified and its public rules verified; see docs/hackathon.md.
A recorded submission video and authenticated portal steps remain unfinished.
No arbitrary-model optimizer or general kernel support is claimed. Browser visual preview of the HTML report was blocked by the local
file URL policy; automated report-content and escaping checks passed.


## Module continuation — November 2, 2026

Executing original Tasks 16–17 through the concrete module plan. API usage remains
25/30: no new generation is needed; reuse the promoted pilot artifacts.
Ruling: begin with flat pure eval-mode FP32 modules — this makes source admission,
live parameter guards and graph boundaries reviewable; unsupported module forms
fall back, so coverage is deliberately narrower than arbitrary PyTorch modules.
Ruling: module optimization consumes promoted artifacts and emits extracted
registered workload specs for the existing function optimizer — avoids a second
kernel evaluator and any unapproved aggregate API spending; unseen signatures
remain fallback-only until separately optimized.


Module Tasks A/B: capture, exact partition, live guards and standalone profiling
implemented. Task C: promoted-artifact rewrite, eight private whole-output cases,
source snapshots and two integrated confirmation gates implemented. Task D CLI
and cached example implemented; final review/evidence/publication pending.
Full suite: 108 passed in 12.44s, no skips. New tests first reproduced missing
interfaces, output/annotated mutation, dtype metadata extraction, invocation and
operator changes, CPU fault classification, keyword fallback and assumed-mode
example-domain validation; each is now green.

First integrated development run `e27a0232849b4431a838f4d417f2c360`:
private cases passed, first whole-module session 1.026×, below 1.05×; no-win and
original fallback retained. Profiling found repeated Pydantic model construction,
module traversal and descriptor inspection in the call guard. Direct metadata
comparison preserves guards and is being measured in a new immutable run.
Ruling: module timing uses the original eager whole module as its explicitly
labeled baseline — function artifacts already beat the verified compiled local
baseline, but no compiled whole-module improvement is claimed; strongest whole-
module compiler comparisons remain a separate research acceptance item.


Fresh module review found no Critical issues and two Important guard gaps.
Final: fixed instance `_call_impl` / `_compiled_call_impl` overrides — new CPU
regressions RED→GREEN, including rejection before invoking an override.
Final: fixed scalar/callable/module attributes read through admitted globals —
`math.pi` change regression RED→GREEN; mutable attribute values and local module
aliases are rejected rather than frozen without guards. Callable code/default
identities are tracked alongside attribute values.
Additional fault-policy regression reproduced an analysis-capture RuntimeError
that did not latch MPS health; the MPS capture boundary now records the fault.
CPU capture errors remain ordinary unsupported outcomes without poisoning GPU health.
Final: Ruling: strongest compiled whole-module comparison remains deferred —
the public baseline is explicitly eager, so these results cannot establish an
advantage over compiling an entire model. Cost if wrong: users may prefer the
compiler; no comparative claim is made.
Final: Ruling: preliminary integrated results are development-only — the final
guard code requires a fresh private gate and two sessions before publication;
old evidence remains retained. Cost if wrong: no final module win is claimed.
No deferred review minors; no second review is required by the execution workflow.


Final review-fix verification: 113 passed in 13.62s, no skips. Both Important
review findings and the MPS analysis fault-latch regression are fixed with
RED→GREEN evidence. The pre-review host-guard optimization rehearsal
`1c9ec52e17a34243be460c2f6462d730` passed eight private cases, two eager whole-module
sessions (1.207× / 1.219×), and post-run value/parameter/shape/gradient checks.
It is retained as development evidence; final guard fixes are remeasured next.


## Final module acceptance

Final run `76212a7153ad4109bd11441f1d6dbb18` passed eight fresh whole-output cases
(max absolute error 1.192e-7), then two original-eager whole-module sessions:
1.220× (lower95% 1.208×) and 1.218× (lower95% 1.204×), 30 pairs each, 60-second
cooldown. Shape is (256,1024), FP32, assumed-domain, 14-core M3 Pro. New values,
live parameter-value updates, changed-shape fallback and gradients passed.
It ran outside the checkout without a key or .env and used zero model requests.
All three development module runs, including the initial no-win and superseded
pre-review promotion, are retained. Candidate identities and bootstrap summaries
were recomputed before curation. Standalone profiling records the same original
graph identity and does not claim additive/exclusive attribution.

The user identified the Meta Global AI Developer Hackathon. Its event page
explicitly lists Muse Wonder and its public Official Rules / Waiver was read.
The submission portal requires sign-in; exact field/video-length requirements
were not visible and no login, terms acceptance or submission was performed.
Prepared copy and evidence mappings are in docs/hackathon.md and
docs/submission-draft.md. Event deadline shown: Nov 4, 2026 9 PM Eastern / 8 PM Chicago.


Final module package checks: wheel and sdist built; fresh wheel installation
outside the checkout passed Metal health, CPU module extraction and no-artifact
fallback through the installed console entry point, with no key/.env and zero
model requests. `results/2026-11-02-module-optimizer/release.json` records source,
wheel/dependency hashes, exact baseline/domain, current hardware and all limits.
Code tree is unchanged from the 113-test review-fix verification.


## Recorded demo continuation — November 2, 2026

Module release published on main at 646179d; CPU CI passed:
https://github.com/ramikamel/metalmind/actions/runs/41544813530.
Continuing original Task 15 with a reproducible recorded backup and a separate
compiled whole-module diagnostic. API generation remains 25/30; no new model
requests are authorized or needed for this work.
Ruling: use a labeled saved-evidence walkthrough plus a real offline correctness
check — reproducible without network/model calls; it cannot be presented as live
kernel generation or a timing recording of the original search.


Compiled whole-module diagnostic completed in two sessions with eight private
cases and a live parameter-update check. Regional rewrite versus eager:
1.123× / 1.112×; versus verified Inductor fullgraph whole module: 0.426× / 0.433×.
This is not a strongest-baseline module win. The default module gate was upgraded
to require every verified available whole-module baseline. Regression first failed
on missing multi-baseline support, then passed; incorrect compiled outputs are
excluded before timing with an explicit unavailable reason.
The real corrected gate `b85f3e29cc1341cca1ed1f42384aa0a3` returned no-win and
kept the original module (1.113× vs eager, 0.426× vs compiled). Earlier eager-only
results remain immutable historical evidence, not current default promotion.

Offline demo source/summary tampering, path confinement, complete denominators,
terminal controls and no-generation CLI tests have passed. Optional GPU checks
now reject custom reference imports (regression RED→GREEN). Rendering tools are
optional extras, with exact versions in requirements-demo.lock.


Release/demo source verification: 127 tests passed in 13.34s, no skips. The
compiled-module diagnostic and actual stricter no-win gate are curated under
results/2026-11-02-module-baselines. Earlier eager-only module results are marked
historical; source and numerical records were preserved unchanged.
Ruling: require all verified available whole-module baselines by default — actual
measurements showed the compiler outperforming the regional rewrite; the cost is
fewer enabled rewrites and compiler setup time, rather than a misleading module win.


Fresh demo/release review found no Critical issues, two Important replay issues,
and one provenance item regraded Important because reproducibility depends on it.
Final: fixed false promotion labels — promoted/enabled state now requires two
complete passing sessions covering exactly the known required baselines, without
duplicates or unknown statuses; direct in-memory gate regressions RED→GREEN.
Final: fixed ratio-preserving raw-sample edits — a checksum manifest bundled with
the code independently binds 137 historical evidence files, never regenerated by
replay. The changed-raw-samples regression failed before the binding was added.
Final: fixed source-commit attribution — recorder preflights the subprocess's actual
package bytes against Git HEAD, and records package location/version/fingerprint;
a dirty-checkout vs committed-source regression failed before provenance helpers existed.
Recorder/renderer code must be committed before producing the release recording.
Final: Ruling: GPU validity and final composition are established by the author's
real GPU tests and rendered-frame checks, rather than the read-only review — cost:
no independent visual/GPU replication is claimed. No deferred review minors.
Rehearsal assets remain ignored; final recording is regenerated after these fixes.


Final review-fix pass: 134 tests passed in 13.84s, no skips. The packaged evidence
manifest, gate-outcome derivation and runtime-vs-Git source provenance tests all
pass. The source checksum file is a release trust anchor; replay never regenerates
it from the input data. Updating curated raw evidence requires an explicit new
freeze/commit, so a different or edited snapshot is rejected.
Ruling: regrade recorder provenance as Important — a checkout SHA must identify
the package actually executing; strict source comparison may reject stale wheels
or uncommitted source, which is preferable to an unrepeatable recording.

## Final demo capture and clean-install verification — November 3, 2026

Recorded ten chapters from commit 209c82e in a temporary working directory
without a model key or local .env. Fresh GPU reuse passed on new values
(maximum absolute error 2.98e-08). Shape and gradient fallback both passed.
The capture ran no timing experiment and made zero Meta generation requests.
Published media are indexed in demo/README.md. The silent 179.96-second video
decoded in full at 1920×1080, 24 fps. Representative frames were inspected.
The ten caption cues, stdout transcript, timestamps, source and file hashes
were checked against the capture and render metadata.

Built wheel and sdist, then installed the wheel in a clean temporary
environment without the video extras. The packaged replay verified all nine
pilot jobs, and the Metal smoke check passed. The installed source fingerprint
matched the recording commit. See demo/clean-install.json for package hashes.


## Frozen comparison continuation — November 3, 2026

With the user's renewed API authorization, ran the four-strategy checked-mode
rehearsal on small FP32 bias/GELU. It completed four jobs, evaluated seven
correct measured candidates, and promoted none. Five Muse Wonder requests
reported 3,285 input and 8,488 output tokens. Raw evidence and hashes are in
results/2026-11-03-strategy-rehearsal. The previous 25-request development
history is separate.

The frozen suite uses 108 checked-mode jobs and a 135-request ceiling, two
requests each for feedback and independent sampling and one for one-shot per
signature/trial. Repeated template tuning is local. The first 12 frozen jobs
have a ceiling of 16 requests. Bounded resume now retains completed outcomes,
recovers terminal results and records interrupted attempts conservatively.
The offline report distinguishes verified search measurements, confirmed
promotion, and matched strategy trials. The latest full local suite passed
143 tests, including real Apple Silicon coverage, before the first batch.

First frozen checkpoint: 12 of 108 jobs, all no-win, no failed jobs.
Nineteen of 20 evaluated candidates passed public correctness and timing;
one candidate failed compilation. The 16 Meta requests reported 10,620
input tokens and 29,728 output tokens. The live and archived offline
analyses matched. The remaining jobs were resumed in bounded batches from
the same source and GPU environment. The local continuation stops on an
evaluator fault or a batch containing a newly failed job.

## Explicit FP16 function capture and offline load cost — November 3, 2026

Task 12 capture now checks explicit FP32 intermediate casts for all three
registered FP16 patterns, and the worker repeats the same admission check.
Negative cases cover early FP16 rounding, a missing final cast, mixed input
dtypes, strings passed as cast targets, invalid call arguments, and discarded
operations that can fail in eager execution. The M3 Pro passed correctness
for all three captured FP16 functions and confirmed FP32 input fallback.
Full local suite: 166 passed. Separate FP16 performance and promotion remain open.

The running FP32 campaign stayed on its original checkout and source. A
large softmax template exhausted its worker deadline, recorded a failed job
with zero Meta requests, and paused. The Metal smoke test passed, and the
campaign resumed from the next job under its original source fingerprint.

Five offline process-level artifact loads measured median total load of
199.63 ms and median Metal compile call of 1.12 ms. At the archived median
workflow saving, the estimated reuse break-even is 6,863 calls for that
measured load. See results/2026-11-03-workflow-amortization.
