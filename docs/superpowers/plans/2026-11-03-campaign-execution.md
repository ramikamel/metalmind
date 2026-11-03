# Campaign execution implementation plan

> Execute inline using the evaluation design in section 14 and Task 13.

**Goal:** Run the frozen comparison in bounded sessions while preserving every job and its request accounting across interruptions.
**Architecture:** Bind the resolved job order and exact package fingerprint to a new manifest. Under an exclusive campaign lock, persist a per-job journal before search. On resume reuse terminal results, mark interrupted searches as failed with their reserved request counts, and advance without replay. Keep the existing verifier and optimizer unchanged.
**Scope:** Nine FP32 signatures, three trials, four strategies. Use checked mode, two proposals/request slots each for feedback, independent and template, and one for one-shot. Up to 135 Meta requests at 4,096 output tokens each. The earlier assumed-domain pilot and the four-job rehearsal are separate.

- [x] Test bounded execution, resume, interrupted job accounting, and the distinction between local proposal slots and Meta requests.
- [x] Add an explicit CLI resume flag and per-invocation job bound. Refuse a changed source fingerprint or concurrent writer.
- [ ] Run the same synthetic protocol end to end, audit outcome accounting, and publish the frozen protocol and measured results available.


The two-request cap prioritizes equal paid budgets and repeat trials. This
measures the effect of the first possible feedback iteration. Three trials
per signature provide descriptive dispersion with limited statistical power.
Review caught unhandled CLI validation and an initialization crash gap. Each
was reproduced by a failing regression and fixed before any full-run traffic.
