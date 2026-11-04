# Live acceptance check from a fresh wheel install

A fresh virtual environment outside the checkout installed the built wheel.
From that environment, the CLI passed the Metal health check, analyzed
medium FP32 bias/GELU, made two Muse Wonder requests, generated and
evaluated Metal candidates, and promoted an artifact. The run generated
its standard HTML report. The artifact passed private verification and both timing confirmations,
measuring **1.566× and 1.575×** versus Inductor fullgraph on the 14-core
Apple M3 Pro in explicit assumed-domain mode.

A separate correctness-only audit verified the new artifact. A fresh
process in the same wheel environment then reloaded it without a model
key or local .env, checked new values, and confirmed shape and gradient
fallback. The offline evidence demo also replayed successfully.

- [Wheel hash and source fingerprint](verification.json)
- [CLI optimization outcome](optimize.json)
- [Fresh artifact verification](fresh-artifact-verification.json)
- [Offline reload of the new artifact](fresh-artifact-reuse.json)
- [Evidence hashes](sha256.json)

The raw directory preserves the emitted run report, event ledger,
candidate sources, confirmation measurements and promoted artifact.
