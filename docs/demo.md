# Demonstration script

1. Explain the supported-function scope and show an ordinary PyTorch bias/GELU function.
2. Run `metalmind doctor --json --smoke` and `metalmind analyze --workload bias_gelu --case medium`.
3. Start a bounded real run: `metalmind optimize --workload bias_gelu --case medium --max-requests 2 --max-candidates 2`.
4. Show saved source, evaluation outcomes, numerical error, and verified baseline identities in the run report.
5. Distinguish numerically valid, search-fast, independently confirmed, and promoted states.
6. If no kernel wins, demonstrate the explained fallback. Do not present a cached or fabricated speedup as live.
7. For a promoted artifact, demonstrate offline reuse with new values and fallback for changed shapes or gradient mode.
8. Report local and integrated workflow latency separately. Label checked versus assumed-domain measurements.

Have a recorded search and a saved report ready for network or timing instability.
Clearly label live, recorded, and cached stages. Never display the local .env.

The public event page lists Muse Wonder; public rules have been checked.
See [event alignment](hackathon.md) for sources and remaining authenticated
portal steps. The script duration here is a proposed presentation length,
not a verified event video limit.


## Saved development demonstration

The [pilot GELU artifact](../results/2026-11-02-fp32-pilot/README.md)
contains source, a 22-case private gate, two independent confirmation sessions,
and raw timing pairs. Use it as an explicitly **cached, assumed-domain** example:

```sh
python scripts/verify_offline_reuse.py results/2026-11-02-fp32-pilot/artifacts/db066f6dfe2e74cefc0c335852229dd2ed6a9058518279eabb7b3a28342caa87
python -m examples.integrated_block --artifact results/2026-11-02-fp32-pilot/artifacts/db066f6dfe2e74cefc0c335852229dd2ed6a9058518279eabb7b3a28342caa87
```

These commands make no model requests. Reuse is enabled only on a matching
environment; another Mac needs explicit revalidation. The integrated command
measures an unchanged matmul plus the guarded activation, checks complete output,
and prints all pairs even when the whole-workflow effect is small or negative.
Do not substitute local GELU gains for the whole-workflow measurement.


## Module extension in the demonstration

1. Show `examples/module_workloads.py`: bias/tanh-GELU followed by unchanged residual math.
2. Run `metalmind analyze-module --factory examples.module_workloads:make_demo`.
3. Show the saved final module run's original/candidate FX graphs and the selected function artifact ID.
4. Explain eight fresh private whole-output cases and the two 30-pair confirmations.
5. Show **1.220× / 1.218× against the original eager whole module**, and the rejected 1.026× draft.
6. Show post-run new-value, live-parameter, shape and gradient checks. State that this replay uses a cached Meta-generated kernel and makes no model requests.

Use the [module evidence](../results/2026-11-02-module-optimizer/README.md) for exact
bounds and source snapshots. Do not label this a compiled whole-model comparison.
A fresh `python -m examples.optimize_module ...` run performs new verification and
both timing sessions; allow its 60-second cooldown, or show the saved result as
explicitly cached evidence during a short presentation.
