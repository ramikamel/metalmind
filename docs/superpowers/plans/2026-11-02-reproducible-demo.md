# Reproducible Demo and Remaining Baseline Plan

> Execute inline using the existing release design (original Task 15), TDD for new application behavior, and one fresh final branch review.

**Goal:** Produce an honest, reproducible recorded backup and close the documented compiled whole-module comparison gap without additional model calls.
**Architecture:** A read-only evidence consumer validates saved artifacts/raw statistics and constructs a labeled demo story. Optional fresh GPU checks use the existing loader. A real CLI recording and a paced video walkthrough use that story; rendering dependencies remain optional. A separate diagnostic compares the existing guarded module against verified Inductor fullgraph.
**Spec:** Original design and implementation plan Task 15; module baseline limitation in docs/module-optimization.md.

## Constraints

- Work in the requested checkout on feat/reproducible-demo.
- Zero additional Meta requests; total generation usage stays 25/30.
- Historical source/evidence remains immutable. New measurements get a new output directory.
- Distinguish saved performance evidence, fresh correctness checks, and presentation pacing.
- Retain all nine pilot outcomes, failed candidates, domain/hardware/baseline labels and whole-module limits.
- Never execute host code stored in an evidence bundle or include .env/credentials in a recording.
- Rendering is optional; CPU-only evidence replay and ordinary tests require no video packages.

## Review focus

1. A compiled loss must block module promotion; both sessions must cover the same required baselines.
2. Changed source, raw samples or outcome labels must not produce trusted replay claims.
3. Evidence paths, references and terminal text must not cause host imports, path escapes or control-sequence injection.
4. Replay must preserve no-win jobs and distinguish saved performance from current correctness.
5. Recording/render provenance must match actual CLI output and never imply presentation pacing is benchmark time.

## Task 1 — Compiled whole-module diagnostic

- [ ] Verify compiled and guarded outputs on eight new values; keep runtime parameters live.
- [ ] Run two sequential paired sessions with 30 pairs/baseline, bootstrap intervals, drift rejection and 60-second cooldown.
- [ ] Record eager and compiled whole-module comparisons separately, including no-win outcomes; do not rewrite the historical eager baseline results.
- [ ] Apply the observed result: default module promotion requires every verified available whole-module baseline, including compiled fullgraph when numerically valid.

## Task 2 — Read-only offline demo

- [ ] RED→GREEN tests: source/summary tampering rejected, nine-job denominator retained, module identity/statistics checked, paths constrained, no network generation, safe terminal text.
- [ ] Add `metalmind demo` with JSON or readable replay; optional fresh artifact correctness/fallback checks and output export.
- [ ] Verify CPU replay outside the checkout without a key or local .env.

## Task 3 — Recorded backup

- [ ] Capture actual CLI stdout/timecodes into a terminal recording.
- [ ] Render a clearly labeled saved-evidence walkthrough and captions from the validated story. Pacing is for readability, never benchmark timing.
- [ ] Inspect representative rendered frames, verify video metadata, record file/source hashes and exact reproduction commands.

## Task 4 — Release verification

- [ ] Run complete tests, build/install checks and fresh final review; fix important findings once with RED→GREEN tests.
- [ ] Update README/demo/submission material with the full baseline context.
- [ ] Scan the full publication tree for credentials, commit, push and verify CI.
