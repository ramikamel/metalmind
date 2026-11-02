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

Eligibility under event-specific Meta model rules remains unverified; resolve it
before presenting a submission as eligible.


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
