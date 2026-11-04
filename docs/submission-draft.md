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

- [Offline evidence showcase](../demo/evidence-showcase.html)

- [Three-minute silent walkthrough](../demo/video/metalmind-evidence-walkthrough.mp4)
- [Terminal recording and captions](../demo/README.md)
- [Function pilot](../results/2026-11-02-fp32-pilot/README.md)
- [Separate FP16 results and reuse audit](../results/2026-11-03-assumed-fp16-pilot/README.md)
- [Fresh-wheel live acceptance check](../results/2026-11-03-clean-wheel-live/README.md)
- [Completed checked-mode campaign](../results/2026-11-03-checked-fp32-full/README.md)
- [Workflow reuse-cost estimate](../results/2026-11-03-workflow-amortization/README.md)
- [Whole-module compiler comparison](../results/2026-11-02-module-baselines/README.md)

The walkthrough shows saved evaluator results from a real Muse Wonder search,
including a valid Metal candidate and a proposal rejected for non-finite output. It then runs
a fresh correctness check using a previously promoted artifact and shows
fallback on a changed shape and in gradient mode.

The assumed-domain FP32 pilot on a 14-core Apple M3 Pro promoted two of nine
signatures. Medium bias/GELU reached 1.547× and 1.553× against PyTorch Inductor
fullgraph in independent confirmations. A separate eager matmul-plus-activation
workflow reached 1.153× in one session. Its median saving was 29.1 µs per call,
which makes the value of reusing a verified artifact visible.

A separate assumed-domain FP16 pilot promoted six of nine signatures
across bias/GELU, RMSNorm and softmax. Medium softmax reached 2.641×
and 2.544× against its fastest verified baseline. Each promoted artifact
dispatched on new in-domain values and used PyTorch fallback on a changed
shape, FP32 inputs and gradient mode.

All 108 jobs in a separate checked-mode comparison fell back to PyTorch.
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
and compiled whole-module gains remain open. The completed checked-mode comparison yielded zero promotions across all
four strategies. It provides no evidence of a feedback advantage at this budget.
The public evidence preserves rejected kernels, unstable windows and no-win jobs.

[Public repository](https://github.com/ramikamel/metalmind)
