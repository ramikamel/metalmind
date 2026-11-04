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
- [Judging rehearsal and live dispatch command](judging-rehearsal.md)
- [Final benchmark results](../RESULTS.md)
- [Original FP32 silent walkthrough](../demo/video/metalmind-evidence-walkthrough.mp4)
- [Terminal recording and captions](../demo/README.md)
- [Fresh-wheel live acceptance check](../results/2026-11-03-clean-wheel-live/README.md)
- [Completed checked-mode campaign](../results/2026-11-03-checked-fp32-full/README.md)
- [Workflow reuse-cost estimate](../results/2026-11-03-workflow-amortization/README.md)
- [Whole-module compiler comparison](../results/2026-11-02-module-baselines/README.md)

The walkthrough shows saved evaluator results from a real Muse Wonder search,
including a valid Metal candidate and a proposal rejected for non-finite output.
Use the rehearsal command to run a fresh GPU check of a promoted artifact and
show fallback on a changed shape, dtype and gradient mode.

On a 14-core Apple M3 Pro, MetalMind promoted four of nine FP32 signatures
and all nine FP16 signatures with caller-guaranteed input bounds. Medium
FP32 bias/GELU measured 1.589× / 1.568× versus Inductor. Medium FP16
softmax measured 2.641× / 2.544× versus eager PyTorch.

Large FP16 RMSNorm measured 1.252× / 1.268× versus Inductor after Muse
Wonder used the earlier kernel source and benchmark feedback to propose a new
schedule. A fixed FP16 attention block reused the softmax artifact and measured
1.058× / 1.067× versus Inductor, the fastest verified whole-block
baseline. Timing included both matrix multiplications.

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
