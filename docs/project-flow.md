# How MetalMind works from input to result

MetalMind takes a supported PyTorch function, searches for a Metal kernel for
one exact tensor signature, and returns either a verified artifact or the
original PyTorch path. Muse Wonder supplies candidate GPU code. The local
controller owns compilation, correctness checks, measurement, and promotion.

Follow the [README system flowchart](../README.md#system-flow) for a visual map,
or use the [recording guide](demo-recording-guide.md) to present the same flow.

## 1. Identify the operation

A developer chooses a registered workload or provides a trusted importable
Python function and example inputs. The analyzer checks the function before
tracing it and matches its FX graph to a supported expression. Supported
expressions include bias plus tanh GELU, RMSNorm, and final-axis softmax.
An unrecognized function stays in PyTorch.

The resulting workload specification records the operation and reference
identity, exact shapes, dtypes, strides, scalar parameters, and numerical
policy. In the demo, medium softmax takes a contiguous FP16 tensor with shape
`(256, 1024)`, computes in FP32, and returns FP16.
[Workload registry](../src/metalmind/workloads/registry.py) · [Analysis](../src/metalmind/analysis.py)

## 2. Search within a budget

The controller reserves request and token capacity, then sends Muse the
workload specification, kernel constraints, and any public feedback from
earlier attempts. Muse returns Metal source and launch dimensions. MetalMind
validates the response schema and source, skips duplicate candidates, and
saves a record of the run.

The search stops when its budget, deadline, or another stop condition is
reached. Each valid candidate competes by its weakest speedup across the
verified PyTorch baselines.
[Search controller](../src/metalmind/agent/controller.py) · [Meta provider](../src/metalmind/agent/meta.py)

## 3. Test and time each candidate on the Mac

A local worker compiles the candidate, dispatches it on the Apple GPU, and
compares the full output to PyTorch on public fixtures. It checks for
unwritten output, input modification, and changes across repeated calls.
Only candidates that pass these checks proceed to timing. Compile errors,
correctness failures, and benchmark measurements can inform the next
proposal while the budget allows it.

The benchmark verifies the available baselines before comparing them.
Depending on the workload, these include eager PyTorch, a comparable
functional implementation, and PyTorch compiled with Inductor. Timing pairs
include the public wrapper, guards, allocation, and GPU completion.
[GPU worker](../src/metalmind/runtime/worker.py) · [Benchmark protocol](benchmarking.md)

## 4. Confirm the best eligible candidate

MetalMind freezes the candidate and checks 22 private fixtures withheld from
search. To promote, it also needs two valid timing sessions. Against every
required verified baseline in both sessions, the median speedup must be at
least 1.05× and the lower bound of the 95% interval must be above 1.00×.

A failed gate leaves the original PyTorch function in use and saves the
reason. A pass writes an artifact with the Metal source, launch parameters,
workload signature, environment fingerprint, and evidence.
[Final gate and artifact export](../src/metalmind/artifacts/export.py)

## 5. Reuse the artifact offline

The loader checks the artifact's integrity, reference identity, and local
hardware and software fingerprint. It then compiles the saved source. On
each call, the wrapper checks the input signature, including shape, dtype,
layout, and device. A gradient-enabled call with a tensor that requires
gradients uses the original PyTorch function.

- **Checked mode** scans the tensor values against the declared numerical
  bounds on every call. A failed value check uses PyTorch.
- **Caller-guaranteed bounds** let the caller opt out of that value scan by
  guaranteeing the bounds. The wrapper still checks the tensor signature.

The value scan is part of measured latency. It helps explain why the checked
FP32 campaign kept PyTorch for all 108 jobs while four FP32 signatures
promoted with caller-guaranteed bounds. Offline reuse makes zero model
requests. [Loader](../src/metalmind/artifacts/loader.py) · [Call guards](../src/metalmind/runtime/execution.py) · [Results](../RESULTS.md)

## How module reuse and campaigns fit

- **Module reuse.** MetalMind can identify supported regions in a flat FP32
  inference module and try replacing them with already-promoted artifacts.
  It verifies and times the whole module before enabling the rewrite. The
  published residual example lost to the compiled whole module and kept
  PyTorch. The fixed FP16 attention benchmark used a manual integration of
  an earlier softmax artifact. [Module workflow](module-optimization.md)
- **Campaigns.** The evaluation runner freezes a workload grid and a request
  budget, then runs the same search and promotion process for each job. It
  retains promotion and fallback outcomes. The tuned prompt sweep tested all
  18 registered FP16 and FP32 signatures. Seven new artifacts also beat
  their earlier MetalMind winners in direct comparisons.
  [Campaign protocol](benchmarking.md) · [Prompt sweep](../results/2026-11-04-muse-prompt-sweep/README.md)

## Follow the demo kernel

The first medium FP16 softmax candidate in the focused repair story looked
fast in search but produced different outputs on repeated inputs. The saved
repair experiment records the diagnosis and the specific synchronization
guidance supplied to Muse. The winning revision passed the private fixtures
and measured 3.098× and 3.183× against eager PyTorch, the fastest verified
baseline in each session. The demo's live reuse audit loads that artifact,
checks fresh values, and observes one Metal dispatch with PyTorch fallback
for the three incompatible calls. [Repair experiment](../results/2026-11-04-muse-prompt-ab/README.md#focused-softmax-repair)
