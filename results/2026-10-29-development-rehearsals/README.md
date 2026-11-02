# Development rehearsals — October 29, 2026

These are development runs, not the frozen 108-job campaign. All four controller
rehearsal outcomes are listed in all-rehearsal-outcomes.json. An earlier tiny
manual generation probe was correct but slower in checked mode.

The selected bias/GELU fixture is FP32, shape (512, 4096), parameter (4096,),
**assumed-domain mode**. This requires caller-guaranteed finite normal/zero inputs
and numerical bounds. It does not describe default checked-mode performance.

The generated artifact passed 22 final numerical fixtures. Session 1 measured
3.130x versus original eager and 1.073x versus compiled fullgraph PyTorch;
session 2 measured 3.064x and 1.076x respectively. See the raw pairs and bootstrap
intervals in confirmation-1.json and confirmation-2.json. This is one shape on
one environment; no cross-chip or all-workload speedup is claimed.

The artifact has also passed a fresh 22-case audit and offline reuse in a fresh
process without a model key or local .env. New input values, changed-shape fallback,
and gradient fallback were checked. The audit's closed fixture descriptors are
retained for reproduction. The initial promotion gate's original random seed
list was not retained; subsequent gates now retain their private descriptors in
ignored local run directories, without exposing them to model feedback.

The source artifact can be audited numerically on a compatible Mac:

```sh
metalmind verify --artifact results/2026-10-29-development-rehearsals/artifact
```

Loading it on another environment remains fallback-only until explicit
revalidation; source portability is not performance portability. The report's
local run/artifact paths describe the original development checkout.

A later hardware-fingerprint strengthening intentionally disables reuse of this
historical artifact. [Fresh revalidation](../2026-11-02-revalidation/README.md)
retains a derived artifact with GPU core identity, new private fixtures, and two
new confirmation sessions. Original measurements above remain unchanged.
