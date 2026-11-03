# MetalMind submission draft

Copy-ready draft. Adapt the sections to the fields and limits shown after sign-in.

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

- [Three-minute silent walkthrough](../demo/video/metalmind-evidence-walkthrough.mp4)
- [Terminal recording and captions](../demo/README.md)
- [Function pilot](../results/2026-11-02-fp32-pilot/README.md)
- [Checked-mode campaign checkpoint](../results/2026-11-03-checked-campaign-checkpoint/README.md)
- [Workflow reuse-cost estimate](../results/2026-11-03-workflow-amortization/README.md)
- [Whole-module compiler comparison](../results/2026-11-02-module-baselines/README.md)

The walkthrough follows a real Muse Wonder search from a rejected proposal
through numerical feedback to a valid Metal kernel. It then runs
a fresh correctness check using a previously promoted artifact and shows
fallback on a changed shape and in gradient mode.

The assumed-domain FP32 pilot on a 14-core Apple M3 Pro promoted two of nine
signatures. Medium bias/GELU reached 1.547× and 1.553× against PyTorch Inductor
fullgraph in independent confirmations. A separate eager matmul-plus-activation
workflow reached 1.153× in one session. Its median saving was 29.1 µs per call,
which makes the value of reusing a verified artifact visible.

The first 12 jobs in a separate checked-mode comparison all fell back to PyTorch.
At module level, the regional rewrite passed eight private correctness cases
and beat eager execution. It lost to the verified compiled whole module, so
MetalMind kept the original module. These cases demonstrate how the gate behaves
when a candidate passes correctness checks but loses on performance.

## Impact and technical execution

The project reduces the manual steps needed to explore local GPU kernels while
making failure ordinary and inspectable. Immutable contracts, independent fixtures,
input integrity, durable budgets, guarded artifacts and two timing confirmations
make proposed gains reviewable. Whole-module rewriting has its own gate, so a
local speedup cannot enable a slower integrated workflow.

## Limits and next work

Current support is a small set of exact pure forward patterns and flat FP32
inference modules. Training, arbitrary nested modules, dynamic shapes, other chips,
and compiled whole-module gains remain open. The published multi-strategy
checkpoint is partial and cannot establish an advantage for feedback.
The public evidence preserves rejected kernels, unstable windows and no-win jobs.

[Public repository](https://github.com/ramikamel/metalmind)
