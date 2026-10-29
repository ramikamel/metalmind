# MetalMind Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This is a planning deliverable, not authorization to start implementation.

**Goal:** Build a local autonomous agent that turns supported PyTorch functions into numerically verified Metal implementations, promotes only confirmed performance wins, and exports guarded reusable callables.

**Architecture:** A Python controller supplies immutable workload specifications and bounded feedback to a Meta model. A local spawned worker compiles candidate Metal source through a version-contained PyTorch MPS backend, verifies it, and performs synchronized comparisons against framework baselines. Versioned artifacts drive both reusable wrappers and the final report.

**Tech Stack:** Python 3.12, pinned PyTorch with MPS and `torch.mps.compile_shader`, Meta Model API through a contained Python client, Pydantic for serialized contracts, PyYAML for configuration, pytest, Typer, Rich, JSON/JSONL, and a small generated HTML report. Add NumPy for statistics; optional plots use Matplotlib. Pin actual dependency versions after capability checks rather than inventing lockfile versions in this plan.

**Spec:** [MetalMind end-to-end system design](../specs/2026-10-29-metalmind-design.md). Read both documents before implementation; the design defines policy and this document defines work order.

**Planning status:** Proposed, October 29, 2026. No application code, package installation, model API call, or GPU benchmark has been performed as part of writing these documents. All commands below are future implementation/acceptance commands.

## Global Constraints

- One person is building the project.
- There is enough time to complete the project; there is no imposed weekend or 48-hour schedule.
- The first release is a **function-level optimizer**. Broader graph analysis and region replacement follow later.
- Meta models drive generation and iterative optimization.
- Pure, deterministic forward functions.
- One tensor output, freshly allocated; input tensors remain unchanged.
- FP32 tensors initially.
- The public function path accepts trusted importable top-level functions; unsupported callables retain their original fallback.
- Checked numerical-domain mode is the public default; assumed-domain mode requires explicit opt-in and separately labeled measurements.
- Exact shapes and exact strides for each optimized signature.
- Tensor parameters remain runtime inputs; their contents must not be frozen into a kernel.
- Python scalar parameters become declared specialization constants and are part of the cache key.
- No autograd support in generated kernels. Calls made with gradient recording enabled and a gradient-requiring input use the original function.
- All serialized contracts carry `schema_version = 1`; loading unsupported future versions fails explicitly.
- Search execution uses one local GPU evaluation at a time.
- Final promotion requires median public-function speedup ≥1.05×, bootstrap interval lower bound >1.00×, and successful repetition in a second confirmation session.
- Files under the project's `sources/` directory are read-only reference material. Create implementation files outside that directory.
- Do not claim first-of-its-kind autonomous Metal optimization or publish hypothetical performance as measured.

## Review Focus

These failure classes require explicit tests in the tasks that own them:

1. **Same shape, different values or tensor parameters:** a reusable kernel must compute from new inputs rather than preserve search answers. Owned by Tasks 3, 5, and 10.
2. **Layout, storage offset, scalar, device, or autograd changes:** an incompatible call must use the original function. Owned by Tasks 2 and 10.
3. **Asynchronous execution, baseline drift, or profiler instrumentation:** results must describe completed comparable work. Owned by Tasks 1, 6, and 13.
4. **Malformed candidate, timeout, cancellation, or crash during persistence:** the run must retain evidence and avoid promoting partial or unhealthy state. Owned by Tasks 4, 7, 8, and 9.
5. **Private gate leakage or reused final evaluation:** search must not adapt to a supposed held-out result. Owned by Tasks 5, 9, and 13.

## 1. Repository layout and module responsibilities

Create this structure in a normal development checkout when implementation begins. The project mirror currently has no application code; no existing implementation patterns are being displaced.

```text
pyproject.toml
README.md
.gitignore
configs/
  default.yaml                 # One-job search settings and budgets
  evaluation.yaml              # Frozen workload jobs and experiment strategies
docs/
  environment.md               # Verified setup, capability matrix, exact wheel pins
  architecture.md              # Maintained runtime overview and support contract
  benchmarking.md              # Timing boundary and reporting methodology
  demo.md                      # Rehearsal and fallback script
src/metalmind/
  __init__.py                  # Supported public API exports only
  contracts.py                 # Versioned serialized data contracts
  config.py                    # Validated settings and environment loading
  workloads/
    registry.py                # Approved workloads, cases, and trusted references
    bias_gelu.py               # Exact PyTorch and oracle semantics
    rmsnorm.py                 # Exact PyTorch and oracle semantics
    softmax.py                 # Exact PyTorch and oracle semantics
    fixtures.py                # Public/private fixture construction
    capture.py                 # Whitelisted FX-to-spec recognition
  runtime/
    environment.py             # Capability probing and fingerprint construction
    metal.py                   # All shader API and dispatch compatibility code
    abi.py                     # Host-owned kernel argument binding and launch rules
    execution.py               # Shared guards, domain checks, allocation, and invocation
    worker.py                  # Spawned evaluator process and protocol
    health.py                  # Known-good GPU probe and fault handling
    lock.py                    # Cross-process evaluation lock
  verification/
    numerics.py                # Equivalence, oracle diagnostics, integrity checks
    policy.py                  # Frozen tolerances and value-domain policies
    final_gate.py              # Private verifier, isolated from model feedback
  benchmarking/
    baselines.py               # Eager, idiomatic, and compiled baseline providers
    timer.py                   # Synchronized paired timing and calibration
    statistics.py              # Paired ratios, bootstrap intervals, drift checks
    promotion.py               # Single promotion policy for all callers
  agent/
    provider.py                # ModelProvider interface and deterministic replay
    meta.py                    # Real Meta API adapter
    prompts.py                 # Versioned prompt construction and feedback views
    candidates.py              # Response parsing, source checks, identity/dedup
    controller.py              # State machine, budgets, archive, stopping
  artifacts/
    store.py                   # Atomic writes, append-only event and source history
    export.py                  # Frozen artifact manifests and evidence checks
    loader.py                  # Offline guarded callable and compiled-object cache
  analysis.py                  # Function analysis and bottleneck hypotheses
  api.py                       # analyze, optimize, and result convenience interface
  reporting/
    summary.py                 # One machine-readable summary from stored evidence
    html.py                    # Escaped static HTML; optional charts
  cli.py                       # Thin CLI handlers calling the public services
examples/
  optimize_function.py         # Small complete developer workflow
  my_workloads.py              # Supported and intentionally unsupported functions
  integrated_block.py          # Whole-workflow measurement with one replaced function
tests/
  unit/                       # CPU-only contract, controller, statistics, storage tests
  gpu/                        # Explicit Apple Silicon integration tests
  live_model/                 # Explicit opt-in endpoint tests
runs/                         # Ignored local run history
artifacts/                    # Ignored generated reusable kernels
```

Keep runtime compatibility, math semantics, model transport, and presentation independent. A change to the report should not change candidate scoring. A provider change should not change the workload specification.

## 2. Shared interfaces and invariants

### 2.1 Types established in Task 2

`Shape = tuple[int, ...]`; tensor dtype names are canonical strings such as `float32`; launch dimensions are three positive integers. Python annotations below refer to these versioned contracts:

- `EnvironmentInfo`, `BackendCapabilities`, `TensorSpec`, `CorrectnessPolicy`.
- `WorkloadSpec`, `AnalysisResult`, `BaselineSummary`.
- `CandidateProposal`, `CandidateSpec`, `EvaluationRequest`, `EvaluationResult`.
- `VerificationSummary`, `TimingConfig`, `TimingSeries`, `ConfirmationResult`.
- `ModelSettings`, `UsageRecord`, `RunConfig`, `RunState`, `RunSummary`.
- `PromotionDecision`, `ArtifactManifest`, `OptimizationResult`.

Supporting contracts are also defined in Task 2: `ProposalContext`, `ProposalResponse`, `FixtureDescriptor`, `ReferenceDescriptor`, `CampaignConfig`, `CampaignManifest`, `CampaignSummary`, `WorkflowComparison`, and `UnsupportedReason`. Runtime-only `Fixture` and `CompiledKernel` live with fixture construction and the backend respectively; they are never serialized into prompts. Follow-on graph contracts are defined only when Tasks 16–17 begin.

`CandidateProposal` is only the model's JSON object. `CandidateSpec` is host-assigned identity plus that proposal. `EvaluationRequest.mode` is one of `search`, `final_verify`, or `confirm`; only the worker and trusted verifier receive fixture handles.

An unsupported analysis has `supported=False`, `workload=None`, and a nonempty reason. An optimization without a winner has `artifact_path=None` and one of `no_win`, `inconclusive`, `budget_exhausted`, `failed`, or `cancelled`. A verified winner has `outcome=promoted` and a complete artifact path.

Source and signature hashes use SHA-256 over canonical UTF-8 JSON or source bytes with defined normalization. Do not use Python's process-randomized `hash()` for artifact identity.

### 2.2 Public service signatures

```python
probe_environment() -> EnvironmentInfo
get_workload(workload_id: str, case_id: str, dtype: str = "float32") -> WorkloadSpec
analyze(fn: Callable, example_inputs: tuple[torch.Tensor, ...]) -> AnalysisResult
optimize(workload: WorkloadSpec, *, config: RunConfig) -> OptimizationResult
load_artifact(path: Path, *, fallback: Callable) -> OptimizedFunction
```

`OptimizedFunction.__call__(*inputs: torch.Tensor) -> torch.Tensor` is the runtime interface. Scalar parameters are bound in the analyzed function and included in its specialization contract; changing them requires a new analysis or an explicit cache miss.

`OptimizationResult.build_callable(fallback: Callable) -> OptimizedFunction` loads a promoted artifact or returns a wrapper that uses only the fallback. The result must not require the caller to distinguish successful search from a valid no-win outcome to obtain correct behavior.

`RunConfig.domain_mode` is `checked` by default or `assume` by explicit choice. Workload value bounds are exactly those in design §6.5; checking mode is included in artifact identity and promotion. A campaign freezes one mode and cannot combine results from both modes into a single speedup claim.

### 2.3 Proposed kernel ABI

Task 1 validates that the selected wheel can implement this contract; Task 2 freezes it:

1. Input buffers in `WorkloadSpec.inputs` order, declared read-only in the source signature.
2. One output buffer allocated by the trusted host.
3. `B` and `D`, passed with an explicit host-controlled unsigned 32-bit scalar policy.
4. One FP32 scalar `epsilon`; use `1e-5` for RMSNorm and `0.0` for contracts that ignore it.

Bias and weight arrays are separate runtime input buffers. Output buffer length, shape, and dtype derive only from the specification. The generated source cannot allocate host memory or select the reference. Each workload has a fixed ABI descriptor; include its version in the semantic hash.

The exact scalar casting and dispatch keywords must be pinned to observed adapter behavior, then covered by integration tests. If the smoke test reveals an incompatible API, update this ABI and the design before implementing dependent components.

### 2.4 Future verification command conventions

Use `python -m pytest tests/unit -q` for CPU-only checks and `python -m pytest tests/gpu -q` for GPU checks on a supported Mac. GPU tests have a registered `gpu` marker and produce an explicit skip reason outside that environment. Live API tests require `METALMIND_RUN_LIVE_MODEL_TESTS=1`; an absent key never causes a normal unit test to make a paid request.

Each implementation task includes a red/green cycle for substantive behavior, a focused verification command, and a logical commit. The commands are instructions to run later; no “tests passed” claim is being made in this planning document.

## 3. Ordered implementation tasks

### Task 1 — Establish direct Metal feasibility and a truthful environment report

**Milestone:** M0. **Dependencies:** none.

**Files:** create `pyproject.toml`, initial `contracts.py` environment types, `src/metalmind/runtime/environment.py`, `src/metalmind/runtime/metal.py`, `src/metalmind/runtime/health.py`, `tests/gpu/test_shader_smoke.py`, `tests/unit/test_environment.py`, and `docs/environment.md`.

**Interfaces:** produce `probe_environment() -> EnvironmentInfo` and `MetalBackend` with `compile(source: str) -> CompiledKernel`, `launch(kernel, inputs, output, launch, constants) -> None`, and `synchronize() -> None`. `CompiledKernel.max_threads_per_threadgroup` is required; any unexposed SIMD width remains `None` rather than guessed.

- [ ] Add a minimal package and explicit GPU test marker while setting up the first executable smoke test.
- [ ] Write a known-good vector transform test that uses real MPS inputs, preallocated output, explicit launch dimensions, a scalar argument, and synchronization.
- [ ] Add tests that compilation of invalid Metal raises a typed compile result and a CPU tensor cannot be passed accidentally to the MPS backend.
- [ ] Run the smoke test before the adapter exists; establish the missing-interface failure.
- [ ] Implement a thin adapter around the public shader API and only the wheel-specific scalar/launch details needed by the test.
- [ ] Probe MPS availability, shader compilation, max threadgroup limit, timed-event availability, and model-independent device health. Verify output visibility across repeated launches.
- [ ] Separately smoke-test the intended Meta endpoint/model with one explicit opt-in small request following its official quickstart. Record redacted access and response-shape facts in setup notes; the durable provider adapter comes in Task 7. Resolve actual hackathon model eligibility before choosing the campaign model.
- [ ] Check PyTorch 2.15 on a compatible OS, with 2.14 as the documented fallback. Choose and pin one wheel and Python environment after the same smoke tests pass.
- [ ] Document installed versions, actual API capabilities, OS requirements, toolchain needs observed on the machine, and unavailable fields. Do not claim full Xcode is required unless the chosen path actually needs it.
- [ ] Run `python -m pytest tests/gpu/test_shader_smoke.py tests/unit/test_environment.py -q`; acceptance is correct synchronized output and explicit diagnostics for unsupported environments.
- [ ] Commit as `feat: establish verified PyTorch Metal execution backend`.

**Gate:** both real Metal execution and intended model access are established. If direct compilation cannot support the ABI, resolve the backend choice now. Do not build the agent around an untested tensor bridge.

### Task 2 — Freeze schemas, configuration, kernel ABI, and signature guards

**Milestone:** M0/M1. **Dependencies:** Task 1.

**Files:** extend and freeze `contracts.py`; create `config.py`, `runtime/abi.py`, `verification/policy.py`, `tests/unit/test_contracts.py`, `tests/unit/test_signature.py`, `tests/gpu/test_scalar_abi.py`, and `configs/default.yaml`.

**Interfaces:** produce all contracts in Section 2; `signature_for(workload: WorkloadSpec) -> str`; `validate_proposal(raw: str) -> CandidateProposal`; `bind_kernel_args(workload, inputs, output) -> tuple`; `matches_signature(workload, inputs) -> bool`.

- [ ] Write serialization tests for a valid workload, candidate, and evaluation result.
- [ ] Assert rejection of unknown schema versions, missing output specifications, negative shapes, invalid launch dimensions, NaN configuration scalars, extra proposal fields, and unsupported dtypes.
- [ ] Assert that changing shape, stride, storage offset, scalar constants, or accumulation policy changes the signature; changing tensor contents does not.
- [ ] Write a GPU scalar ABI test whose output depends separately on `B`, `D`, and `epsilon`, using values that distinguish integer and floating-point casts.
- [ ] Run these tests to establish their failure before implementation.
- [ ] Implement immutable Pydantic contracts with forbidden extra fields at model boundaries, canonical hashing, explicit enums, and validation of cross-field invariants.
- [ ] Implement configuration loading with environment variables for secrets and schema validation for budgets and paths. Store redacted settings in run artifacts.
- [ ] Freeze the proposed tolerances and budget defaults from the design. The private fixture plan is not a model-visible configuration object.
- [ ] Make the 30-minute job wall limit include a five-minute finalization reserve. Freeze 10,000 paired bootstrap resamples, percentile bounds at 2.5% and 97.5%, and the minimum 60-second confirmation-session cooldown.
- [ ] Freeze the exact value domains: bias/GELU inputs and bias in `[-10,10]`; RMSNorm inputs in `[-100,100]` and weights in `[-10,10]`; softmax inputs in `[-80,80]`; all finite, with nonzero subnormal inputs excluded. Default to checked mode.
- [ ] Run `python -m pytest tests/unit/test_contracts.py tests/unit/test_signature.py tests/gpu/test_scalar_abi.py -q`; acceptance includes stable serialization across a new process.
- [ ] Commit as `feat: define immutable workload and candidate contracts`.

### Task 3 — Implement the three workload contracts and independent fixtures

**Milestone:** M1. **Dependencies:** Task 2.

**Files:** create `workloads/registry.py`, `workloads/bias_gelu.py`, `workloads/rmsnorm.py`, `workloads/softmax.py`, `workloads/fixtures.py`, `tests/unit/test_workload_semantics.py`, and `tests/gpu/test_reference_functions.py`.

**Interfaces:** produce `get_workload(...) -> WorkloadSpec`, `resolve_reference(reference_id: str) -> Callable`, `make_public_fixtures(workload) -> list[Fixture]`, and verifier-only `make_private_fixtures(workload, private_seed_set) -> list[Fixture]`. A `Fixture` contains trusted input tensors and a category; it never enters a model prompt directly.

- [ ] Write small hand-computable tests for bias broadcasting, tanh GELU selection, RMS epsilon placement, RMS absence of mean subtraction, and softmax normalization axis.
- [ ] Add CPU FP64 mathematical checks for the three FP32 fixtures, including nonuniform bias and weight values.
- [ ] Test that two different parameter tensors with identical metadata produce the expected distinct outputs.
- [ ] Add irregular dimensions, zero rows, close maxima, common softmax shifts, and large but supported finite values.
- [ ] Implement references with explicit dtype and cast policies; register the nine proposed FP32 jobs.
- [ ] Implement deterministic public fixtures and separate private fixture descriptors. Generate inputs on CPU first so randomness does not depend on GPU RNG behavior.
- [ ] Verify private seed descriptors are absent from serialized model feedback and public fixture manifests.
- [ ] Run `python -m pytest tests/unit/test_workload_semantics.py tests/gpu/test_reference_functions.py -q`.
- [ ] Freeze reference hashes and policy version in the registry manifest.
- [ ] Commit as `feat: add explicit workload semantics and independent fixtures`.

**Gate:** a reference/policy disagreement is a setup defect, not a model failure. Resolve it before candidate search.

### Task 4 — Persist run history and execute candidates in controlled workers

**Milestone:** M1. **Dependencies:** Tasks 1–3.

**Files:** create `.gitignore` entries for generated runs, artifacts, and local secret configuration; create `artifacts/store.py`, `runtime/worker.py`, `runtime/lock.py`, `tests/unit/test_artifact_store.py`, `tests/unit/test_worker_protocol.py`, and `tests/gpu/test_worker_execution.py`.

**Interfaces:** produce `ArtifactStore.create_run(config, environment) -> Path`, `append_event(event) -> None`, `write_candidate(candidate) -> Path`, `evaluate(request: EvaluationRequest) -> EvaluationResult`, and `check_health() -> bool`.

- [ ] Write tests that a candidate source snapshot is immutable and a failed manifest write leaves the last complete manifest readable.
- [ ] Test a truncated final JSONL record: recover the valid prefix and mark an interrupted event; do not silently erase earlier events.
- [ ] Test timeout, worker crash, cancellation, incompatible response schema, and lock contention with fake worker processes.
- [ ] Implement spawn-based workers that receive only validated request objects, resolve trusted registry code, and write bounded structured responses.
- [ ] Remove model credentials from the worker environment; do not put credentials in source, events, or exception text.
- [ ] Implement the cross-process GPU evaluation lock, separate compile and execution deadlines, source-size limit, output limits, and post-fault health gate.
- [ ] Use atomic file replacement for manifests and immutable content-addressed source snapshots; the event stream supplies history, while manifests supply current state.
- [ ] Verify a real known-good candidate works in a spawned process without transferring live MPS tensors.
- [ ] Run `python -m pytest tests/unit/test_artifact_store.py tests/unit/test_worker_protocol.py tests/gpu/test_worker_execution.py -q`.
- [ ] Commit as `feat: add durable runs and controlled GPU evaluation workers`.

### Task 5 — Build correctness checking and the separate final verification gate

**Milestone:** M1. **Dependencies:** Tasks 3–4.

**Files:** create `verification/numerics.py`, `verification/final_gate.py`, `tests/unit/test_numerical_policy.py`, `tests/gpu/test_candidate_correctness.py`, and debug-only broken shader fixtures under `tests/gpu/fixtures/`.

**Interfaces:** produce `verify_candidate(candidate, workload, fixtures) -> VerificationSummary`, `run_final_gate(candidate, workload, private_plan) -> VerificationSummary`, and `public_feedback(result) -> dict`. `public_feedback` has no access to private fixture values, seed IDs, or expected outputs.

- [ ] Write threshold tests at, below, and above `atol + rtol * abs(reference)`, including near-zero reference values and nonfinite candidate outputs.
- [ ] Create intentionally incorrect kernels: constant output, skipped tail, swapped bias indexing, altered input, unwritten element, and nondeterministic/racy reduction where practical.
- [ ] Assert each incorrect candidate is rejected with an informative public category and cannot proceed to timing.
- [ ] Implement comparisons against the approved PyTorch reference plus separate high-precision diagnostic metrics.
- [ ] Implement sentinel-filled outputs, input integrity checks, repeated-run checks, and safe debug tail guard allocations. Logical output size remains host-owned.
- [ ] Add explicit finite-domain handling and a checked-mode path; do not infer value-domain compliance from tensor metadata.
- [ ] Implement a final private gate that can be consumed only once per frozen candidate and evaluation campaign. Failure closes promotion for that job.
- [ ] Add optional Shader Validation execution with mode recorded, and ensure it is not enabled by the performance confirmation path.
- [ ] Run `python -m pytest tests/unit/test_numerical_policy.py tests/gpu/test_candidate_correctness.py -q`.
- [ ] Commit as `feat: reject incorrect kernels and isolate final verification`.

**Gate:** no invalid kernel is assigned a speed score; all required output elements must pass, not a percentage.

### Task 6 — Implement synchronized baselines, paired timing, and promotion statistics

**Milestone:** M1/M3. **Dependencies:** Tasks 3–5.

**Files:** create `runtime/execution.py`, `benchmarking/baselines.py`, `benchmarking/timer.py`, `benchmarking/statistics.py`, `benchmarking/promotion.py`, `tests/unit/test_statistics.py`, `tests/unit/test_promotion.py`, `tests/gpu/test_timing.py`, and `docs/benchmarking.md`.

**Interfaces:** produce `collect_baselines(workload, environment) -> BaselineSummary`, `measure_pair(baseline, candidate, config: TimingConfig) -> TimingSeries`, `summarize_pairs(series) -> ConfirmationResult`, and `decide_promotion(verification, confirmations) -> PromotionDecision`.

Also produce `build_execution_callable(workload, kernel, *, fallback, domain_mode) -> Callable` in `runtime/execution.py`. Evaluator timing and offline loading must use this same guard/check/allocation/dispatch path; the exporter must not benchmark a privileged shortcut.

- [ ] Write unit tests using synthetic paired samples: exact 1.00× tie, 1.04× insufficient gain, 1.10× stable gain, and noisy data whose lower interval crosses 1.00×.
- [ ] Test the rule that both independent sessions must pass and that missing or failed correctness evidence always prevents promotion.
- [ ] Write GPU timing tests that deliberately enqueue work before synchronization and verify the timer awaits completion. Use ordering assertions and broad workload scaling, not brittle microsecond thresholds.
- [ ] Implement eager, idiomatic, and attempted compiled baseline providers. Baseline availability records include exact callable configuration and failure reasons.
- [ ] Verify each baseline's semantics with the workload's numerical policy before it can be the primary comparator.
- [ ] Implement at least 25 warmups, block-size calibration, balanced seeded order, ten search pairs, thirty confirmation pairs, raw sample persistence, and optional secondary event timing.
- [ ] Implement the five-pilot-block stability check from design §12.3, 25-call warmup increments, and 200-call cap; persistent instability produces an inconclusive measurement.
- [ ] Ensure primary callable timing includes guards and fresh output allocation on the candidate side; inputs remain resident on MPS on both sides.
- [ ] Include finite/range/subnormal checking and its synchronization cost in checked mode; test out-of-domain contents call the original function. Measure assumed-domain mode only under its declared preconditions and label it separately.
- [ ] Implement seeded paired bootstrap, median paired speedup, interval bounds, control-baseline drift detection, and whole-window rerun metadata.
- [ ] Document the amortized timing boundary and separately record cold compile and model costs.
- [ ] Run `python -m pytest tests/unit/test_statistics.py tests/unit/test_promotion.py tests/gpu/test_timing.py -q`.
- [ ] Commit as `feat: compare kernels with synchronized paired benchmarks`.

**Gate:** a complete manual evaluator can now compile, reject, measure, and classify a candidate without any language model.

### Task 7 — Integrate Meta generation behind a replaceable provider interface

**Milestone:** M2. **Dependencies:** Tasks 2, 4–6.

**Files:** create `agent/provider.py`, `agent/meta.py`, `agent/prompts.py`, `agent/candidates.py`, `tests/unit/test_meta_adapter.py`, `tests/unit/test_prompt_boundaries.py`, and `tests/live_model/test_meta_generation.py`.

**Interfaces:** produce `ModelProvider.propose(context: ProposalContext) -> ProposalResponse`, `MetaProvider(settings: ModelSettings)`, and `ReplayProvider(responses: list[ProposalResponse])`. `ProposalResponse` contains the raw normalized response, parsed proposal or parse error, usage, request identity, and model provenance.

- [ ] Write fake-transport tests for a valid response, malformed JSON, extra fields, truncation, empty content, 401, 429, transient 5xx, and request timeout.
- [ ] Assert authentication failure is not retried blindly; transient retries stop after the configured maximum and cannot exceed run budgets.
- [ ] Assert prompts include mathematics, exact signature, public feedback, budgets, and actual known capabilities; they omit keys, private fixtures, expected output arrays, and unrestricted execution instructions.
- [ ] Implement the documented direct Meta transport with `muse-wonder-1.0` as the proposed main model and explicitly configured `muse-spark-1.3` as an access/comparison alternative.
- [ ] Keep the endpoint, API surface, reasoning settings, maximum output, and model IDs in validated configuration. Do not silently change models on an error.
- [ ] Implement strict candidate parsing and source policy checks, then attach host-generated IDs and immutable workload identity.
- [ ] Record reported usage, including reasoning tokens where provided. Missing usage/cost metadata remains null; it never becomes zero.
- [ ] Run the CPU tests, then perform one explicitly enabled small live generation against the selected endpoint. Save a redacted capability result with actual response shape.
- [ ] Run `python -m pytest tests/unit/test_meta_adapter.py tests/unit/test_prompt_boundaries.py -q`; run the live test only with the opt-in flag and configured key.
- [ ] Commit as `feat: generate constrained Metal candidates with Meta models`.

**Gate:** the real model can return a candidate that reaches the evaluator. A correct but slower candidate is an acceptable transport milestone.

### Task 8 — Implement the autonomous controller and bounded feedback loop

**Milestone:** M2. **Dependencies:** Tasks 4–7.

**Files:** create `agent/controller.py`, add feedback views to `agent/prompts.py`, and create `tests/unit/test_controller.py`, `tests/unit/test_budgets.py`, and `tests/gpu/test_live_loop_replay.py`.

**Interfaces:** produce `SearchController.run(workload, config) -> RunSummary`, `SearchController.cancel() -> None`, and `SearchController.resume(run_path) -> RunSummary`. `RunSummary` identifies the frozen search incumbent and is not itself a promotion decision.

- [ ] With `ReplayProvider`, test this sequence: malformed proposal → compile failure → correctness failure → correct slow result → correct promising result → stop.
- [ ] Assert the controller supplies feedback automatically and never requires a developer to edit candidate source between attempts.
- [ ] Assert identical source/launch duplicates do not trigger GPU reevaluation; repeated malformed responses still consume request budget.
- [ ] Assert invalid candidates never enter the numerical-valid archive and no candidate mutates the workload specification.
- [ ] Implement explicit state transitions, best-three archive, concise failure memory, deterministic parent selection, source deduplication, and event emission.
- [ ] Implement request, candidate, wall-time, token, spend, and stagnation limits. Reserve the next request's allowed output budget before dispatch; document possible estimate error when exact provider token pricing is unavailable.
- [ ] Reserve the final five minutes of the default 30-minute job for finalization, clamp in-flight deadlines to remaining time, and test that an unfinished final gate yields budget-exhausted fallback rather than promotion.
- [ ] Implement cancellation at safe boundaries and worker termination when needed; preserve an interrupted state without pretending the evaluation completed.
- [ ] Implement resume checks for schema, workload, config, source hashes, and environment. Timing confirmation is not reusable across an incompatible environment.
- [ ] Run `python -m pytest tests/unit/test_controller.py tests/unit/test_budgets.py tests/gpu/test_live_loop_replay.py -q`.
- [ ] Commit as `feat: add bounded autonomous kernel optimization loop`.

### Task 9 — Freeze winners, enforce final gates, and export evidence

**Milestone:** M3. **Dependencies:** Tasks 5, 6, and 8.

**Files:** create `artifacts/export.py`, extend `agent/controller.py`, and create `tests/unit/test_finalization.py` and `tests/gpu/test_finalize_artifact.py`.

**Interfaces:** produce `finalize_run(summary: RunSummary) -> OptimizationResult`, `export_artifact(candidate, verification, confirmations) -> ArtifactManifest`, and `verify_artifact_evidence(manifest) -> None`.

- [ ] Test that source changes after selection invalidate final verification and prevent export.
- [ ] Test that a private gate failure closes the job with no promoted artifact and never reappears in model feedback.
- [ ] Test each final outcome: promoted, no win, inconclusive, budget exhausted before a valid candidate, cancelled, and failed.
- [ ] Freeze the selected candidate and its hashes, run sixteen private-seed fixtures plus structured checks, and then collect fresh confirmation sessions.
- [ ] Apply the same promotion function used in statistics tests; never duplicate promotion rules in the CLI or report.
- [ ] Export only complete immutable evidence: source, host ABI version, guards, environment, policies, baseline identity, verification summary, raw timing references, and provenance.
- [ ] Keep incomplete or rejected finalization records in the run directory; do not put them in the promoted-artifact namespace.
- [ ] Run `python -m pytest tests/unit/test_finalization.py tests/gpu/test_finalize_artifact.py -q`.
- [ ] Commit as `feat: export only independently confirmed optimization artifacts`.

### Task 10 — Load offline artifacts with deterministic guards and fallback

**Milestone:** M3. **Dependencies:** Task 9.

**Files:** create `artifacts/loader.py`, `api.py`, `__init__.py`, `tests/unit/test_loader_guards.py`, and `tests/gpu/test_artifact_reuse.py`.

**Interfaces:** produce `load_artifact(path, *, fallback) -> OptimizedFunction`, `OptimizedFunction.__call__(*inputs) -> torch.Tensor`, `OptimizationResult.build_callable(fallback)`, and `revalidate_artifact(path: Path, *, fallback: Callable, config: RunConfig) -> OptimizationResult`.

- [ ] Use a spy fallback to test shape, dtype, stride, storage offset, device, scalar-specialization mismatch, and autograd conditions.
- [ ] Test empty tensors and aliasing between inputs remain outside the optimized signature, and checked mode falls back for nonfinite, out-of-range, or nonzero subnormal contents.
- [ ] Test `requires_grad=True` under normal gradient mode uses the original function and produces correct gradients; under explicit no-grad execution it may use a matching forward artifact.
- [ ] Test corrupted source or evidence hashes, unsupported schema, and incompatible environment disable the optimized path before dispatch.
- [ ] Load the same artifact in a new process with no API key and run new random inputs plus new nonuniform bias/weight tensors.
- [ ] Implement trusted host wrapper allocation, metadata guards, compiled-object caching by full identity, offline operation, and fallback-only results.
- [ ] Reuse `build_execution_callable` from Task 6. Verify the supplied fallback's reference/semantic fingerprint matches the artifact before enabling its kernel.
- [ ] Keep incompatible-environment loads fallback-only. Explicit revalidation runs a fresh numerical gate, new strongest-baseline collection, and both confirmation sessions, then writes a derived artifact if it wins; a correctness-only audit does not enable promotion.
- [ ] On synchronous load/compile failure, invalidate the artifact and report the reason. Do not implement an unsafe promise to recover invisibly from asynchronous GPU faults.
- [ ] Verify primary latency again through the actual exported wrapper rather than a special evaluator shortcut.
- [ ] Run `python -m pytest tests/unit/test_loader_guards.py tests/gpu/test_artifact_reuse.py -q`.
- [ ] Commit as `feat: reuse verified Metal artifacts through guarded PyTorch callables`.

**Gate:** the first complete vertical slice now exists: selected function → real model search → verification → timing → finalization → offline reuse or explicit fallback.

### Task 11 — Recognize supported user functions and produce useful analysis

**Milestone:** M4. **Dependencies:** Tasks 3, 6, and 10.

**Files:** create `workloads/capture.py`, `analysis.py`, `examples/my_workloads.py`, `tests/unit/test_capture.py`, `tests/unit/test_reference_resolution.py`, and `tests/gpu/test_function_analysis.py`.

**Interfaces:** produce `capture_function(fn, example_inputs) -> WorkloadSpec | UnsupportedReason`, `resolve_user_reference(descriptor) -> Callable`, and public `analyze(...) -> AnalysisResult`.

- [ ] Write tests that recognize importable top-level bias/GELU, RMSNorm, and softmax functions with the exact declared semantics.
- [ ] Reject lambdas, local closures, stateful callables, mutable captures, tensor-valued globals, input mutation, data-dependent control flow, unknown operations, and ambiguous broadcasts with explicit reasons.
- [ ] Assert exact GELU cannot be mislabeled tanh GELU, and centered LayerNorm cannot be mislabeled RMSNorm.
- [ ] Assert capture fingerprints include source/module digest, recognized semantics, default scalar values, immutable scalar captures, and reference identity.
- [ ] Implement a narrow FX pattern recognizer and canonical expression serializer. Trace failures are unsupported results, not invitations for the model to guess the semantics.
- [ ] Resolve the user's actual function in the worker by its trusted importable descriptor, verify its fingerprint and normalized semantics again, and time that actual callable as the original baseline.
- [ ] Gather measured function latency, tensor sizes, operation summary, estimated intermediate traffic, and clearly labeled bottleneck hypotheses.
- [ ] Keep function-level capture separate from whole-model graph rewriting. Do not add graph replacement infrastructure in this task.
- [ ] Run `python -m pytest tests/unit/test_capture.py tests/unit/test_reference_resolution.py tests/gpu/test_function_analysis.py -q`.
- [ ] Commit as `feat: analyze supported importable PyTorch functions`.

**Gate:** a user-defined supported function can enter the same optimization path as bundled fixtures without changing its reference semantics or disguising a reconstructed graph as its original baseline.

### Task 12 — Extend the whole pipeline to FP16 with explicit FP32 intermediates

**Milestone:** M5, optional for the initial release. **Dependencies:** Tasks 5, 10, and 11.

**Files:** modify workload references, `verification/policy.py`, `workloads/capture.py`, `runtime/abi.py`, and candidate prompts; create `tests/unit/test_fp16_semantics.py` and `tests/gpu/test_fp16_pipeline.py`.

**Interfaces:** extend existing `WorkloadSpec` contracts with `input_dtype=float16`, `accumulation_dtype=float32`, and `output_dtype=float16`; add no competing optimization API.

- [ ] Write tests that distinguish FP16 intermediate rounding from the chosen FP32-intermediate policy, especially bias addition and reductions.
- [ ] Define all three FP16 reference functions by casting tensor inputs to FP32, executing the declared expression, then casting the single output to FP16.
- [ ] Add recognition patterns for those explicit casts; do not treat every FP16 PyTorch expression as equivalent automatically.
- [ ] Add FP16 versions of all nine jobs, supported-domain boundaries, overflow-sensitive fixtures, and nonuniform runtime parameters.
- [ ] Enforce the separate frozen FP16 tolerances and dtype-aware signatures; an FP32 artifact cannot match an FP16 call.
- [ ] Verify numerical correctness, output/input integrity, actual wrapper latency, offline reuse, and fallback across dtype changes.
- [ ] Run `python -m pytest tests/unit/test_fp16_semantics.py tests/gpu/test_fp16_pipeline.py -q`.
- [ ] Commit as `feat: support explicitly specified FP16 optimization workloads`.

**Gate:** FP16 is reported as supported only if the complete pipeline passes; successfully compiling `half` arithmetic is insufficient.

### Task 13 — Freeze an evaluation campaign and test the contribution of feedback

**Milestone:** M6. **Dependencies:** Tasks 6, 8–11; Task 12 only for an FP16 campaign.

**Files:** create `evaluation.py`, `configs/evaluation.yaml`, `examples/integrated_block.py`, `tests/unit/test_evaluation_accounting.py`, `tests/unit/test_campaign_freeze.py`, and `tests/gpu/test_integrated_block.py`.

**Interfaces:** produce `run_campaign(config: CampaignConfig) -> CampaignSummary`, `freeze_campaign(config) -> CampaignManifest`, and `measure_workflow(original, optimized, fixtures) -> WorkflowComparison`.

- [ ] Write accounting tests that include a promoted job, intentional fallback, compile failure, correctness failure, and interrupted run in the correct denominators.
- [ ] Assert changing a workload, budget, reference, tolerance, or model setting invalidates a frozen campaign identity.
- [ ] Implement nine required FP32 jobs and three independent trials per strategy, with seed and configuration tracking. Make FP16 a distinct campaign rather than silently doubling the denominator.
- [ ] Freeze domain mode for each campaign. A separate assumed-domain campaign may establish conditional speedups; it cannot be presented as default checked-mode performance.
- [ ] Implement eager/idiomatic/compiled baseline collection, one-shot generation, simple template-parameter search, and feedback-driven search.
- [ ] Add an optional equal-budget independent-sampling comparison and a feedback-withheld ablation. If omitted, report the resulting limitation on causal claims about feedback.
- [ ] Keep search/public fixture seeds distinct from the final private gate; publish final seed descriptors only after the campaign is closed if needed for independent reproduction.
- [ ] Run strategies on the GPU sequentially, interleave campaign order, and record baseline drift and invalidated complete windows.
- [ ] In the integration example, keep all other operations unchanged. Compare the entire workflow output and synchronized latency, including real wrapper overhead.
- [ ] Produce per-job rows, success rates, confidence intervals, geometric aggregates, costs, candidate failure distributions, and all-job versus winner-only summaries.
- [ ] Run `python -m pytest tests/unit/test_evaluation_accounting.py tests/unit/test_campaign_freeze.py tests/gpu/test_integrated_block.py -q`.
- [ ] Run a small rehearsal campaign before the full frozen campaign; treat rehearsal measurements as development data.
- [ ] Commit as `feat: evaluate optimization quality with complete reproducible campaigns`.

**Gate:** both a positive result and a no-win result produce complete interpretable evidence. The report cannot hide failed jobs.

### Task 14 — Ship the CLI, report, and complete developer example

**Milestone:** M7. **Dependencies:** Tasks 9–11 and 13.

**Files:** create `cli.py`, `reporting/summary.py`, `reporting/html.py`, `examples/optimize_function.py`, `tests/unit/test_cli.py`, `tests/unit/test_reporting.py`, and `tests/gpu/test_cli_journey.py`.

**Interfaces:** expose `doctor`, `analyze`, `optimize`, `verify`, `revalidate`, `report`, and `evaluate` CLI commands. Produce `build_run_summary(run_path) -> dict` and `render_report(summary, output_path) -> Path`.

- [ ] Write CLI tests using fake services for unavailable MPS, unsupported function, exhausted budget, promoted artifact, and no-win outcome.
- [ ] Assert status labels distinguish numerically valid, search-fast, final-verified, confirmed, and promoted states.
- [ ] Build the summary solely from stored manifests/events/raw measurements. HTML must escape generated source, compiler errors, and model text.
- [ ] Display the exact baseline identity, timing boundary, dtype, shape, errors, tolerance version, interval, model provenance, budget usage, and live/cached/replay label.
- [ ] Show failures and no-win outcomes as ordinary readable results. Render missing metrics as unavailable rather than zero.
- [ ] Implement `verify` as a correctness audit that does not enable an incompatible artifact. Implement `revalidate` as explicit numerical plus baseline/performance confirmation on the current environment, producing a derived artifact only if promotion passes.
- [ ] Provide a short example that analyzes an importable function, optimizes it, builds a callable, runs new values, and demonstrates an incompatible-shape fallback.
- [ ] Run `python -m pytest tests/unit/test_cli.py tests/unit/test_reporting.py tests/gpu/test_cli_journey.py -q`.
- [ ] Commit as `feat: expose MetalMind through a usable CLI and evidence report`.

**Optional afterward:** add a local read-only live dashboard that consumes the same event log. A separate web stack, public hosting, user accounts, and a remote GPU service are outside this task.

### Task 15 — Validate the full story and prepare the hackathon release

**Milestone:** M7. **Dependencies:** Tasks 1–11, 13–14; include Task 12 only when claiming FP16 support.

**Files:** finish `README.md`, `docs/environment.md`, `docs/architecture.md`, `docs/benchmarking.md`, `docs/demo.md`, package metadata, release manifest, and selected shareable campaign evidence.

**Interfaces:** produce a documented package installation, executable example, reproducibility recipe, and demonstrable claim-to-evidence mapping.

- [ ] Run the complete unit suite and the GPU suite on the pinned target environment. Run live model smoke tests explicitly rather than from ordinary CI.
- [ ] Reproduce one complete run from a clean environment: diagnose → analyze → real model generation → evaluator feedback → finalization → report → offline artifact reload.
- [ ] Verify both a promoted path, when one exists, and the no-win/fallback path. Rehearse shape changes, new tensor values, and gradient mode.
- [ ] Build the package and install it in a fresh environment to catch missing package data and console entry points.
- [ ] Record exact commands, runtime pins, model configuration, source/evidence hashes, and campaign IDs. Include a CPU-only CI path; GPU performance checks remain dedicated and opt-in.
- [ ] Confirm which Meta models and submission materials are eligible under the actual event rules. Record source and dependency attribution, including any borrowed template or benchmark code.
- [ ] Write the three-to-five-minute demo script from the design and prepare an honestly labeled recorded backup and previously verified artifact.
- [ ] Tie every slide or README performance claim to a specific confirmation result. Use “target” for unachieved goals and disclose strongest-baseline availability.
- [ ] Remove credentials from shareable material, while retaining numerical and timing data needed to assess the claims.
- [ ] Commit as `docs: prepare reproducible MetalMind release and demonstration`.

**Release gate:** apply the design's definition of done, and distinguish engineering completion from success of the performance hypothesis.

## 4. Follow-on work after the function-level release

These are planned extensions, not dependencies of the initial hackathon result. Begin each only after the prior milestone has a working artifact and measurements.

### Task 16 — Profile supported regions in a larger PyTorch module

**Milestone:** M8, analysis portion. **Dependencies:** function-level release.

**Proposed files:** `graph/capture.py`, `graph/regions.py`, `graph/profile.py`, `tests/unit/test_region_partition.py`, and `tests/gpu/test_module_analysis.py`.

**Interfaces:** `analyze_module(module, example_inputs) -> ModuleAnalysis`, containing supported pure regions, graph guard requirements, measured or estimated latency attribution, and exclusion reasons.

- [ ] Freeze a small representative module with known pure candidate regions and unchanged library operations.
- [ ] Test capture failures, shared inputs, multiple consumers, unsupported branches, parameters, and mutation boundaries.
- [ ] Partition only semantically supported regions. Retain unsupported portions unchanged.
- [ ] Measure the original whole module and selected regions; do not sum overlapping profiler times as if they were independent GPU durations.
- [ ] Rank regions using observed time share and measured local optimization opportunities.
- [ ] Produce a report with candidate function extraction and all assumptions visible.

**Acceptance:** the analysis identifies useful opportunities without claiming to have optimized the model yet.

### Task 17 — Replace validated regions with guarded artifacts

**Milestone:** M8, integration portion. **Dependencies:** Task 16 and existing artifact pipeline.

**Proposed files:** `graph/rewrite.py`, `graph/guards.py`, `tests/unit/test_rewrite_guards.py`, and `tests/gpu/test_module_replacement.py`.

**Interfaces:** `optimize_module(module, example_inputs, config) -> OptimizedModuleResult` with graph, parameter, shape, dtype, and execution-mode guards.

- [ ] Test region boundaries, intermediate lifetimes, multiple graph uses, output structure, and fallback to the original module.
- [ ] Extract eligible regions into the existing function optimizer; do not implement a second evaluator.
- [ ] Replace only promoted artifacts and preserve parameter tensors as runtime inputs.
- [ ] Check complete module outputs against the original for private values and all declared graph guards.
- [ ] Confirm whole-module latency including graph dispatch and guard overhead.
- [ ] Disable a rewrite if the integrated workflow does not improve, even when a local function benchmark did.

**Acceptance:** a whole module obtains a confirmed measurable improvement with correct fallback and explicit supported graph boundaries.

### Later research backlog, in recommended order

| Extension | New problem it must solve | Required evidence before claiming support |
|---|---|---|
| Cross-chip tuning | Device-dependent schedules and artifact compatibility | Independent correctness and performance confirmation on each chip |
| Bounded dynamic shape buckets | Valid domains beyond exact signatures | Private in-domain shapes and boundary fallback tests |
| Multi-pass kernels | Scratch memory, dependencies, and lifetimes | Complete dispatch-plan correctness and public-wrapper timing |
| More numerical domains | Nonfinites, subnormals, extreme values | Explicit semantics and adversarial oracle tests |
| Backward generation | Gradient semantics and integration with autograd | Gradient checks and whole training-step comparisons |
| Richer search algorithms | Whether complexity improves outcomes | Budget-matched comparisons against the original controller |
| Model training or fine-tuning | Dataset quality and reward validity | Held-out workloads, independent evaluator, and actual improvement over inference-only search |

## 5. Test and acceptance matrix

| Requirement | Main tests | Owning tasks |
|---|---|---|
| Real direct Metal execution | Shader smoke, scalar ABI, device health | 1–2 |
| Immutable semantics | Contract serialization, signature hashing, oracle semantics | 2–3 |
| Numerical failure rejection | Tail, bias indexing, constant output, mutation, nonfinite output | 5 |
| Safe controller failure paths | Malformed response, retries, timeout, cancellation, resume | 4, 7–8 |
| No hidden baseline disadvantage | Baseline semantics and availability records | 6, 13 |
| Completed comparable timing | Synchronization, balanced paired order, drift window policy | 6 |
| Correct promotion | Threshold, confidence interval, two sessions, no private leakage | 6, 9 |
| Reuse with new values | New random inputs and nonuniform parameter tensors | 3, 5, 10 |
| Guards and fallback | Shape, dtype, stride, storage offset, scalar and gradient mode | 2, 10 |
| Actual user reference | Importability and source/semantic fingerprint recheck | 11 |
| FP16 policy | Explicit casts and separate numerical campaign | 12 |
| Complete evaluation | All-job denominator, failures, ablation and aggregate tests | 13 |
| Usable product flow | CLI journey, report evidence consistency, clean install | 14–15 |
| Broader model optimization | Region guards and whole-module correctness/performance | 16–17 |

## 6. Dependency order and solo working method

```text
1 → 2 → 3 → 4 → 5 → 6
                    6 → 7 → 8 → 9 → 10 → 11
                                          11 → 12 (FP16 extension)
                                          11 → 13 → 14 → 15
                                                  release → 16 → 17
```

The arrows summarize sequencing; each task's dependency list is authoritative. Work sequentially as a solo developer. Build the manual evaluator before the model loop, the complete one-function path before extra workloads, and evidence before presentation polish.

At each milestone, retain one small known-good end-to-end example. A failure should be localizable to capture, semantics, transport, compilation, correctness, timing, or reuse. Avoid developing all layers simultaneously and discovering their interface mismatches only at demo time.

## 7. Resource and cost planning

### Local resources

- One compatible Apple Silicon Mac is sufficient for the initial release.
- Keep model inference remote during timing; local inference on the same GPU distorts measurements and consumes shared memory.
- Measure one GPU evaluation at a time; CPU-only analysis and report generation do not need a GPU worker.
- Bound candidate source, live tensors, scratch, log size, model requests, and search duration even without a hackathon deadline.
- Separate regenerable runs from curated shareable results. No large model weights are a required repository dependency.

### Model cost model

For each provider request, compute cost from reported billable input/output tokens, cached-token metadata when available, and a dated pricing snapshot. Keep actual provider invoices/usage authoritative. Estimated cost is labeled estimated.

For scale, nine FP32 jobs × three trials × thirty proposals permits up to **810 candidate evaluations** for the main search strategy, before ablations. That is a campaign upper bound, not a requirement to spend the full budget or a claim of expected success.

An illustrative thirty-request job with 6,000 billed input tokens and 2,000 billed output tokens per request totals 180,000 input and 60,000 output tokens. Apply the selected model's current prices; reasoning tokens belong in billed output when the provider accounts for them that way. Longer prompts, retries, and deeper reasoning change the total. Do not mistake visible generated source length for all billed output.

Run a single-family pilot and inspect measured request count, tokens, compile time, verification time, and benchmark time before launching the full campaign. The pilot informs a deliberate cap; it must not be mixed into held-out evaluation results.

## 8. Scope changes and stop/go decisions

| Observation | Action |
|---|---|
| Shader API cannot support trustworthy execution | Settle a single alternative backend before expanding the agent |
| Model generates invalid syntax repeatedly | Improve ABI/examples and error feedback; retain a simple correct template for evaluator diagnosis |
| Correct candidates are slower | Examine actual baseline and removable traffic; do not relax correctness or hide the baseline |
| Only device timing improves | Optimize wrapper/dispatch/allocation or reject promotion |
| Private verification fails | End that job's promotion; use a new private gate for a new campaign |
| Primary workload is already highly optimized | Add another explicit fused function with a frozen contract; record the scope change before search |
| Graph capture becomes a time sink | Release registered supported function examples first, while keeping public arbitrary-function claims narrow |
| FP16 remains numerically ambiguous | Finish and report FP32; keep FP16 as an unshipped extension |
| Dashboard work distracts from the optimizer | Use the CLI and static report until the core evidence is complete |

## 9. Implementation handoff

The first execution action is **Task 1**, not a dashboard, general compiler backend, or advanced search framework. Verify the direct PyTorch-to-Metal path and write the actual environment facts down.

Before implementing, review the design's proposed choices: supported function restrictions, finite value domains, numerical tolerances, model selection, timing boundary, and promotion policy. Any change to those must update both the design and affected tests before search results are collected.

The planning documents themselves contain no promised speedup and no fabricated experiment. The first credible performance claim will come from a frozen source artifact with numerical evidence and independent timing confirmation.
