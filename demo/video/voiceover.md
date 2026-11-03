# Optional narration script

Silent video; record your own narration if desired.

## 00:00:00.000 — MetalMind
A Meta-powered agent for verified Apple GPU kernel optimization. Model proposes Metal -> compile -> verify -> benchmark -> feedback. Host code owns fixtures, numerical decisions and promotion. No model generation is performed in this playback.

## 00:00:18.000 — The complete pilot
9 signatures: 2 promoted; 7 no-win. 18 model requests; all outcomes stay in the denominator. Bias + tanh GELU, RMSNorm and stable row softmax. Target: Apple M3 Pro, 14 GPU cores. Checked mode is the public default; these gains assume the input domain.

## 00:00:36.000 — The real feedback loop
Candidate 1 (67a32c18f3f0): valid — 10 public numerical cases passed. Candidate 2 (7b12d24b03bf): incorrect — nonfinite output. These are saved evaluator results from actual Meta proposals. Search results are provisional; private and timing gates follow.

## 00:00:54.000 — bias_gelu / large
Shape (512, 4096); 22 private numerical cases passed. Session 1: 1.093x vs Inductor fullgraph; lower 95% 1.086x. Session 2: 1.097x vs Inductor fullgraph; lower 95% 1.089x. Public-call timing includes guards, allocation, dispatch and completion. Kernel source SHA-256: cdff502221bed6f03786...

## 00:01:12.000 — bias_gelu / medium
Shape (256, 1024); 22 private numerical cases passed. Session 1: 1.547x vs Inductor fullgraph; lower 95% 1.534x. Session 2: 1.553x vs Inductor fullgraph; lower 95% 1.543x. Public-call timing includes guards, allocation, dispatch and completion. Kernel source SHA-256: 4a6da691021af4135e24...

## 00:01:30.000 — Meta-generated Metal
Immutable source: 4a6da691021af4135e241e39... Full source is preserved with its numerical and timing evidence.

## 00:01:48.000 — Whole-module gains need context
A flat residual-activation module has its own correctness/timing gate. Historical eager-only sessions: 1.220x / 1.218x. A local kernel gain is not automatically a strongest whole-model gain.

## 00:02:06.000 — The compiler counterexample
Session 1 vs eager whole module: 1.123x Session 1 vs compiled whole module: 0.426x Session 2 vs eager whole module: 1.112x Session 2 vs compiled whole module: 0.433x Below 1.0x means the regional rewrite is slower than that baseline. Module promotion now requires every verified available baseline. Current module gate: no_win; optimized rewrite enabled: False.

## 00:02:24.000 — Offline reuse check
Outcome: passed New input values matched the original PyTorch function. Changed shapes and gradient mode used the original fallback. Maximum absolute error: 2.98e-08.

## 00:02:42.000 — Scope and next frontier
Exact shapes and supported pure forward patterns on the recorded chip. No arbitrary nested-model, training or cross-chip speedup claim. The full multi-strategy campaign has not been run. This is saved evidence, not live generation or a recording of search latency. Source, raw pairs, intervals and no-win outcomes are public.
