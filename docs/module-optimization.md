# Module profiling and guarded replacement

MetalMind can analyze a conservative inference-module subset and reuse its
confirmed function artifacts inside the graph. This path makes **zero model
requests**. It extracts the same immutable registered workload specs accepted
by the function optimizer; new signatures still need function-level optimization.

## Supported boundary

Start with a flat `torch.nn.Module` in eval mode, ordinary contiguous zero-offset
FP32 tensors, positional tensor inputs, registered parameters/buffers, direct
scalar attributes, and allowlisted pure Torch operations. Child modules, hooks,
control flow, mutable Python state, instance invocation overrides, local Python
module aliases, device transfers, and mutation are rejected before tracing.
This does not support arbitrary models or nested `nn.Linear` module trees.

Within that boundary, exact bias/tanh-GELU, RMSNorm and softmax expressions are
recognized. Other admitted pure operations remain in PyTorch. Shared boundary
inputs and multiple uses of the region output are supported; an outside consumer
of an internal region value excludes the region. Dependent regions preserve their
original graph uses and nested tensor output structures.

## Analyze a trusted module

A factory is explicit user-selected Python code returning `(module, inputs)`.
Never point it at untrusted downloaded code.

```sh
metalmind analyze-module --factory examples.module_workloads:make_demo
metalmind analyze-module --factory examples.module_workloads:make_demo --measure
```

CPU metadata analysis is available through `make_cpu_demo`; Metal replacement
requires MPS. The report records exact workload specs, graph identity, exclusions,
estimated input bytes, and optional standalone timing samples. Measured regions
rank ahead of unmeasured estimates. Standalone times are **not exclusive GPU time**,
can exceed whole-module latency, and must not be summed as attribution.

A selected region's `analysis.regions[i].workload` can be passed to the existing
`metalmind.optimize(workload, config=RunConfig(...))` API. That is a separate,
explicitly budgeted generation action; module analysis/reuse does not call it.

## Reuse a promoted artifact

```sh
metalmind optimize-module --factory examples.module_workloads:make_demo \
  --artifact results/2026-11-02-fp32-pilot/artifacts/db066f6dfe2e74cefc0c335852229dd2ed6a9058518279eabb7b3a28342caa87 \
  --assume-domain

python -m examples.optimize_module \
  --artifact results/2026-11-02-fp32-pilot/artifacts/db066f6dfe2e74cefc0c335852229dd2ed6a9058518279eabb7b3a28342caa87 \
  --assume-domain
```

Checked mode is the default. The cached pilot artifact is assumed-domain, so
explicit opt-in is required. Callers then guarantee the declared domain for the
inputs of **every replaced region**, including intermediates. Both example inputs
and private gate values are checked before timing, even in assumed mode.

The existing artifact loader verifies source/evidence identity, current hardware/
software fingerprint, numerical contract and selected domain mode. A mismatch
leaves the original module enabled. There is no second kernel compiler/evaluator.

## Complete-module gate

The provisional rewrite is verified on eight private input cases. Complete
outputs, nested structures, input/parameter integrity, and region input domains
are checked. Runtime parameter values remain live. This is additional evidence
on top of each function artifact's 22-case final gate.

Promotion then requires two whole-module sessions separated by 60 seconds, at
least 30 balanced pairs per session, median gain at least 1.05× and bootstrap
lower 95% bound above 1.00×. The benchmark includes graph dispatch, all guards,
allocation, and unchanged operations. A slow or unstable integrated rewrite is
disabled even if its local kernel was promoted.

The whole-module baseline is **the original eager module**. A compiled whole-
module comparison has not been performed; local function-vs-Inductor gains must
not be presented as whole-model compiler gains.

Calls use the optimized path only when input signatures, module code, live
parameter identity/metadata, scalar/global attribute dependencies, invocation
behavior, hooks, execution mode and relevant flags still match. New values and
in-place parameter-value updates remain supported. Shapes/layouts, gradients,
training, autocast, keyword calls, parameter replacement and code/state changes
use the original callable. Use the returned callable within `torch.inference_mode()`;
it is not an `nn.Module` replacement for training or serialization.

## Evidence and limits

Each saved run contains analysis, original/candidate FX source, function artifact
IDs, immutable host-implementation snapshots, private seed descriptors, raw
confirmation pairs, final enabled/no-win status and zero-request provenance.
Incomplete runs never return an enabled rewrite. Module runtime/capture faults
latch GPU health; `doctor --smoke` must pass before more GPU evaluation.

[Measured module development runs](../results/2026-11-02-module-optimizer/README.md)
retain the initial no-win and the final reviewed result. These measurements cover
one residual-activation module and signature on one chip. Broader graph coverage,
compiled whole-module baselines, training, and the full strategy campaign remain
research work.
