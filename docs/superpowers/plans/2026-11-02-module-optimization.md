# Module Profiling and Guarded Replacement Implementation Plan

> Use superpowers:executing-plans inline, with TDD and one fresh final review.

**Goal:** Complete original Tasks 16–17 for a conservative, measurable inference-module subset.
**Architecture:** Admit a flat pure module before FX tracing; match the existing exact registered function patterns inside its graph. Profile standalone regions without additive attribution. Reuse promoted artifacts through the existing loader, guard the whole graph, verify new complete-module values, and enable only after two whole-module timing confirmations.
**Tech stack:** Existing Python/PyTorch/Metal runtime, frozen report contracts, dataclass execution state, existing paired timing and artifact loader. No new dependencies.
**Spec:** [approved project design](../specs/2026-10-29-metalmind-design.md), original plan Tasks 16–17.

## Constraints and rulings

- Work in the requested checkout on `feat/module-optimizer`; no new API calls in this implementation run.
- Initial module support: flat eval-mode modules, ordinary FP32 tensors/parameters/buffers, positional tensor inputs, pure allowlisted Torch operations. Unsupported source, child modules, hooks, control flow, or mutation yields a reason and the original fallback.
- Exact supported bias/tanh-GELU, RMSNorm and softmax regions only. Internal nodes with outside consumers are excluded; shared boundary inputs and multiple root consumers remain valid.
- Parameters remain live tensor inputs. Metadata/identity replacement, scalar/global/code changes, hooks, training, gradients, autocast, or input signature changes trigger whole-module fallback.
- Checked domain remains default. Assumed-domain artifact reuse requires explicit opt-in; no implicit upgrade.
- Reuse original evaluator/loader for local kernels; do not create a second kernel correctness or compilation path.
- A graph rewrite is provisional until complete outputs pass private new-value cases and whole-module paired timing passes 1.05× median / lower 95% >1 in two sessions separated by 60 seconds.
- Standalone region times are not exclusive GPU attribution, must not be summed, and can exceed whole-module time.
- Model generation remains an explicit separately budgeted function-level step. Extracted registered workload specs feed that existing optimizer; this module path consumes its promoted artifacts.

## Review focus

1. Shared internal nodes versus shared boundaries: never erase an outside user's value.
2. Parameter replacement, in-place content changes, monkeypatched code, globals, hooks and autocast: preserve original behavior or fall back.
3. Multiple region dependencies and nested outputs: preserve graph uses and complete output structure.
4. Changed signatures or assumed-domain mismatch: do not silently use an incompatible artifact.
5. A local kernel win without an integrated gain: return original module and retain no-win evidence.

## Task A — Capture, partition, and guard metadata

Files: `graph/capture.py`, `graph/regions.py`, `graph/guards.py`, `graph/contracts.py`, unit tests.
- [x] Write failing tests for exact region extraction, shared intermediates/boundaries, mutable/control-flow rejection before tracing, parameter identity/content changes and output structure.
- [x] Admit source and graph purity; snapshot function/global/scalar/module/parameter metadata.
- [x] Match existing normalized reference patterns and extract callable regions with runtime boundary inputs.
- [x] Unit suite green; commit capture/partition deliverable.

## Task B — Useful module analysis and profiling

Files: `graph/profile.py`, public API, unit/GPU tests.
- [x] Test CPU metadata analysis and unavailable measurement labels; GPU-test complete module and extracted-region agreement.
- [x] Measure whole original module and standalone regions under the GPU lock, with synchronization, warmup, raw samples and drift rejection.
- [x] Rank by measured standalone latency when available, otherwise estimated bytes. Preserve exclusions and graph identity.
- [x] Expose `analyze_module(module, example_inputs, measure=False)` and JSON-ready `ModuleAnalysis`.

## Task C — Guarded artifact rewrite and integrated gate

Files: `graph/rewrite.py`, unit/GPU tests.
- [x] Test no-artifact fallback, opt-in domain mode, incompatible artifacts, shared uses, changed code/state/input/gradient modes and no integrated win.
- [x] Rewrite only matched promoted artifacts using existing `load_artifact`; preserve runtime parameters and all other operations.
- [x] Verify complete outputs for eight private new-value cases and input/state integrity. Check region domains even for assumed-mode private tests.
- [x] Use existing paired statistics for two integrated confirmation sessions; no enabled rewrite on an unstable/no-win window.
- [x] Persist analysis, artifact IDs, raw confirmations, numerical outcomes, no-win reasons and immutable graph/guard identity.

## Task D — CLI, example, evidence and release

Files: CLI, `examples/module_workloads.py`, `examples/optimize_module.py`, docs and curated results.
- [x] Add explicit trusted factory-based analysis/reuse commands; offline by default, no hidden generation.
- [x] Run real cached-artifact module journey, new values, parameter changes, shapes and gradient fallback.
- [x] Run full CPU/GPU suite and package checks; one fresh whole-branch review, one tested fix pass.
- [x] Publish all integrated outcomes, including a legitimate no-win result, without a whole-model/general-module claim.
