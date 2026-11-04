# MetalMind

A Meta-powered experimental agent that generates Metal kernels for supported
PyTorch functions, checks them numerically on Apple Silicon, and measures them
against verified framework baselines. Kernels are reused only after independent
correctness and performance confirmation; otherwise PyTorch remains the fallback.

## Run locally

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Put your Meta key in the ignored local `.env`:

```dotenv
MODEL_API_KEY="your-key"
```

```sh
metalmind doctor --json --smoke
metalmind analyze --workload bias_gelu --case medium
metalmind optimize --workload bias_gelu --case medium --max-requests 2 --max-candidates 2
metalmind report --run runs/<run-id>
python -m pytest -q
```

Optimization makes billable Meta model requests. The example caps the request
count at two. No API calls occur during ordinary tests or campaign preview.

## What works

- FP32 bias plus tanh GELU, RMSNorm, and stable final-axis softmax contracts.
- Registered FP16 variants with explicitly FP32 intermediate arithmetic.
- Restricted capture of trusted importable FP32 and explicitly cast FP16 functions.
  See [FP16 examples](examples/fp16_workloads.py) for the accepted cast order.
- Meta Responses API proposals, bounded feedback-driven search, source deduplication,
  cancellation boundaries, durable request accounting, and guarded resume.
- Spawned GPU evaluation, full-output numerical comparison, input integrity,
  unwritten/tail detection, repeated execution, and private final verification.
- Eager, equivalent functional, and compiled framework comparisons, paired timing,
  raw samples, bootstrap intervals, and two-session promotion.
- Conservative flat FP32 module-region analysis, standalone profiling, and guarded
  artifact replacement with a separate whole-module gate.
- Offline guarded artifact loading, fallback, numerical audit, explicit revalidation,
  complete run reports, and a frozen sequential campaign runner.

```sh
metalmind optimize --function examples.my_workloads:bias_gelu --workload bias_gelu --case medium --max-requests 2
metalmind evaluate --suite configs/evaluation.yaml
```

The campaign command previews **108 checked-mode jobs** without contacting Meta.
The frozen suite allows up to **135 Meta requests**. Add `--execute --max-jobs 12`
to run a bounded batch. Continue the same campaign with
`--execute --resume --max-jobs 12` from the same checkout. See the
[campaign protocol](docs/benchmarking.md) for accounting and recovery.

## Benchmark results

Measured on a **14-core Apple M3 Pro GPU**. The speedups below show both
confirmation sessions against the fastest verified PyTorch baseline.

| Benchmark | Final result |
| --- | --- |
| FP16 functions with caller-guaranteed bounds | **9 of 9 signatures promoted**. Medium softmax reached **2.641× / 2.544×** vs eager PyTorch. |
| FP32 functions with caller-guaranteed bounds | **4 of 9 signatures promoted**. Medium bias/GELU reached **1.589× / 1.568×** vs Inductor. |
| Default checked FP32 campaign | **0 of 108 jobs promoted**. PyTorch handled every case. |
| Complete fixed FP16 attention block | **1.058× / 1.067×** vs Inductor, including both matrix multiplications. |
| FP32 residual module | PyTorch retained. The rewrite lost to Inductor. |

[All 18 workload results and the complete workflow comparisons](RESULTS.md)

## Module workflow

The module path reuses confirmed function artifacts and makes no model requests.
It supports a conservative set of flat, pure FP32 inference modules; child modules
and unsupported behavior retain the original callable.

```sh
metalmind analyze-module --factory examples.module_workloads:make_demo --measure
```

The module gate now requires a gain against **every verified available baseline**,
including compiled whole-module PyTorch. The residual example was faster than
eager execution but slower than Inductor fullgraph, so its rewrite is disabled.
This preserves the original callable and exposes the compiler result.

[Usage and guards](docs/module-optimization.md) ·
[Compiler comparison and no-win decision](results/2026-11-02-module-baselines/README.md)

## Offline demo

```sh
metalmind demo
metalmind demo --gpu-check --output runs/demo.json
```

Replay verifies saved source identities and raw statistics, includes all nine
pilot outcomes, and makes **zero model requests**. `--gpu-check` adds fresh
correctness and fallback checks. Performance numbers come from recorded experiments.
See the [three-minute walkthrough](demo/video/metalmind-evidence-walkthrough.mp4)
and [terminal recording](demo/README.md). Use `--evidence-root` to point an installed
CLI at this checkout’s matching `results` snapshot.

## Numerical domain and performance boundaries

Checked mode is the public default. It verifies finite, bounded, nonsubnormal
input contents and includes that cost in timing. For tiny inputs these checks can
make a correct generated kernel substantially slower than PyTorch.

`--assume-domain` is an explicit opt-in for callers who guarantee the documented
input bounds. Those results are labeled separately and do not describe default
checked-mode speed. Bias/GELU inputs and parameters are bounded by 10; RMSNorm
inputs by 100 and weights by 10; softmax inputs by 80. Zero is allowed.

Initial artifacts specialize exact shapes, dtype, strides, scalar constants, and
runtime environment. Training calls, mismatching shapes/layouts, and incompatible
environments use the original function. Arbitrary or nested models, general graph
replacement, backward generation, and cross-chip portability remain future work.

The local evaluator is a fault-containment boundary, not a secure sandbox for
hostile GPU programs. After a GPU fault, further evaluations stop until a
successful `metalmind doctor --smoke` health check.

## Project records

- [System design](docs/superpowers/specs/2026-10-29-metalmind-design.md)
- [Implementation plan](docs/superpowers/plans/2026-10-29-metalmind.md)
- [Progress and verified results](docs/progress.md)
- [Architecture](docs/architecture.md)
- [Environment](docs/environment.md)
- [Measurement methodology](docs/benchmarking.md)
- [Offline evidence showcase](demo/evidence-showcase.html)
- [Judging rehearsal](docs/judging-rehearsal.md)
- [Presenter guide and read-aloud script](docs/presenter-guide.md)
- [Demo and reproduction guide](docs/demo.md)
- [Hackathon rules alignment and prepared submission](docs/hackathon.md)

PyTorch is pinned to 2.15.0. GPU checks skip explicitly when MPS is unavailable.
Use `requirements-dev.lock` for the tested dependency snapshot. The event lists
Muse Wonder among its models; [rule alignment](docs/hackathon.md) distinguishes
confirmed public requirements from the authenticated submission steps. The full
comparison campaign has its own frozen protocol and result ledger.
