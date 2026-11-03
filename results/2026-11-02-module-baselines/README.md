# Compiled whole-module comparison — November 2, 2026

The regional rewrite **did not beat compiled whole-module PyTorch**. This closes
the earlier explicitly unmeasured baseline gap and changes the default module
promotion policy. The function-level gains against compiled function baselines
remain separate, unchanged experiments.

Same residual-activation module, FP32 `(256, 1024)`, explicit assumed-domain mode,
14-core M3 Pro. The compiler baseline passed eight fresh input cases plus a runtime
parameter-update case (maximum absolute error 2.384e-7); the guarded candidate
passed its own eight-case whole-output check. No model requests were made.

| Session | Regional rewrite vs eager whole module | Regional rewrite vs Inductor fullgraph whole module |
| --- | ---: | ---: |
| 1 | 1.123× | 0.426× |
| 2 | 1.112× | 0.433× |

Ratios are baseline latency divided by rewritten latency; below 1.0 means the
rewrite is slower. Both sessions retain 30 balanced pairs per baseline, 10,000
bootstrap resamples, and a 60-second cooldown. See [report.json](report.json)
for exact intervals, all samples, baseline availability and verified environment.
The original eager-only 1.220× / 1.218× result remains historical evidence, not
an advantage over the compiler.

## Corrected default gate

The current module optimizer verifies and compares **every available whole-module
baseline**, including Inductor fullgraph when numerically valid. It requires both
sessions to pass every baseline. Unavailable/incorrect compiler paths are recorded;
they are never silently presented as a completed compiler comparison.

A fresh end-to-end run with that rule returned **no_win**, `enabled=false`, and
retained the original callable. Its first session measured 1.113× versus eager
and 0.426× versus compiled; failure stopped promotion immediately. The
[strongest-gate](strongest-gate/result.json) directory retains the new source/policy
identity, private verification and raw nested confirmation records (format v2).

```sh
python -m scripts.probe_module_baselines \
  --artifact results/2026-11-02-fp32-pilot/artifacts/db066f6dfe2e74cefc0c335852229dd2ed6a9058518279eabb7b3a28342caa87
```

This diagnostic uses an owned demo module and zero model requests. Source snapshots
and private seeds are retained. Current module API gates are conservative: a
locally winning Metal function is not enough to enable an inferior integrated path.
