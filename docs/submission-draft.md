# MetalMind submission draft

Prepared text; not submitted. Adapt field lengths to the authenticated portal.

## One-sentence description

MetalMind uses a Meta model to generate, test and refine Metal GPU kernels for
supported PyTorch operations on Apple Silicon, promoting only numerically verified,
independently confirmed performance gains.

## Problem and intended users

Apple Silicon makes GPU acceleration accessible to PyTorch developers, but
hardware-specific kernel tuning still requires expertise in memory access,
parallel reductions and Metal execution. MetalMind explores whether a model can
perform that iterative engineering loop with concrete compiler, correctness and
benchmark feedback. The target user is a developer optimizing known pure inference
functions on a local Mac.

## How Meta tooling is central

`muse-wonder-1.0` proposes actual Metal source and launch geometry. The trusted
host compiles it, checks complete outputs against immutable PyTorch semantics,
rejects invalid kernels, and returns observed errors/timing to the next proposal.
The model drives candidate generation; it never supplies executable host Python,
fixture answers, benchmark scores or acceptance decisions.

## Working demonstration

Show one stored generation/feedback sequence and its source, then the correctness
and strongest available function-baseline comparisons. The nine-case FP32 pilot
promoted 2 signatures and retained 7 no-win outcomes. Reuse a promoted artifact
offline with new values, then demonstrate shape and gradient fallback. Finally
show conservative module-region extraction: the final residual-activation module
passed eight private cases. Explain that an eager-only gain was superseded by
a compiled whole-module comparison: the compiler won, and the stricter default
gate now disables the regional rewrite. Show that no-win decision alongside the
function-level wins to demonstrate baseline-aware acceptance.
Clearly label any stored search as cached/recorded; do not imply live API activity.

Use the exact figures and bounds in the linked results, including assumed-domain
labels. Function-level Inductor comparisons and original-eager whole-module
comparisons are different experiments. No general or compiled whole-model speedup
is claimed.

## Impact and technical execution

The project reduces the manual steps needed to explore local GPU kernels while
making failure ordinary and inspectable. Immutable contracts, independent fixtures,
input integrity, durable budgets, guarded artifacts and two timing confirmations
make proposed gains reviewable. Whole-module rewriting has its own gate, so a
local speedup cannot enable a slower integrated workflow.

## Limits and next work

Current support is a small set of exact pure forward patterns and flat FP32
inference modules. Training, arbitrary nested modules, dynamic shapes, other chips,
compiled whole-module gains and a full multi-strategy evaluation remain unproven.
No claim is made that feedback causally outperforms one-shot or independent sampling.
The public evidence preserves rejected kernels, unstable windows and no-win jobs.

Repository: https://github.com/ramikamel/metalmind
