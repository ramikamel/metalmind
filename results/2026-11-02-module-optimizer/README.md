# Guarded module development evidence — November 2, 2026

**Historical eager-only policy.** A later [compiled whole-module comparison](../2026-11-02-module-baselines/README.md)
found the compiler faster; the current default gate disables this rewrite. The
measurements below remain unchanged records of the earlier eager comparison.

The final reviewed rewrite of `ResidualActivation`, FP32 input `(256, 1024)`,
measured **1.220×** and **1.218×** versus the **original eager whole module**.
Lower 95% bounds were **1.208×** and **1.204×**. Each session retained 30 raw pairs,
10,000 paired bootstrap resamples, and the complete guard/dispatch boundary;
sessions were separated by 60 seconds. This is **assumed-domain** performance
on the 14-core M3 Pro environment, not a compiled whole-model comparison.

The module contains bias/tanh-GELU plus unchanged residual multiplication and
addition. It reuses the previously promoted medium GELU kernel; the module path
made **zero model requests**. Session generation usage remains 25 of 30 authorized.

Eight fresh private whole-output cases passed (max absolute error 1.192e-7).
New values, in-place bias updates, changed-shape fallback and gradient fallback
also passed. The final example ran outside the checkout without an API key or
local `.env`; `offline-context.json` records that provenance.

| Development stage | Outcome | Whole-module sessions |
| --- | --- | --- |
| Initial guard draft | no-win; original retained | 1.026×; below 1.05× gate |
| Profiled guard, before final review fixes | superseded development promotion | 1.207× / 1.219× |
| Final reviewed guards | promoted | 1.220× / 1.218× |

These are successive engineering rehearsals of one workload, not independent
campaign trials; no aggregate success rate is inferred. The first draft did not
retain host implementation snapshots. Later runs retain immutable graph and host
source identity. The final guard implementation closes the review findings around
instance invocation overrides and mutable module-valued global dependencies.

Primary run: [76212a7153ad4109bd11441f1d6dbb18](76212a7153ad4109bd11441f1d6dbb18/result.json).
Its directory retains original/candidate FX source, artifact IDs, private fixture
seeds, numerical results, both confirmations, implementation snapshots and
post-run checks. Candidate hashes and saved summaries were recomputed from those
snapshots and raw samples before publication. FX source is stored as JSON strings
to preserve the framework-generated whitespace exactly; host implementation
snapshots remain Python source.

[standalone-profile.json](standalone-profile.json) measures the original module and
selected extracted region separately. These observations are not exclusive GPU
attribution and must not be summed; the profile uses the same original graph identity.

```sh
python -m examples.optimize_module \
  --artifact results/2026-11-02-fp32-pilot/artifacts/db066f6dfe2e74cefc0c335852229dd2ed6a9058518279eabb7b3a28342caa87 \
  --assume-domain
```

Reuse requires matching artifact hardware/software identity and explicit domain
opt-in. Each new module instance is privately checked and independently timed
before a rewrite is enabled; no saved module result silently grants promotion.
See [support and guard details](../../docs/module-optimization.md).
