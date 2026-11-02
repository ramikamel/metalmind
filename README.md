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
- Restricted capture of trusted importable FP32 functions matching those exact patterns.
- Meta Responses API proposals, bounded feedback-driven search, source deduplication,
  cancellation boundaries, durable request accounting, and guarded resume.
- Spawned GPU evaluation, full-output numerical comparison, input integrity,
  unwritten/tail detection, repeated execution, and private final verification.
- Eager, equivalent functional, and compiled framework comparisons, paired timing,
  raw samples, bootstrap intervals, and two-session promotion.
- Offline guarded artifact loading, fallback, numerical audit, explicit revalidation,
  complete run reports, and a frozen sequential campaign runner.

```sh
metalmind optimize --function examples.my_workloads:bias_gelu --workload bias_gelu --case medium --max-requests 2
metalmind evaluate --suite configs/evaluation.yaml
```

The campaign command previews **108 jobs**. Add `--execute` only when you intend
to run the configured campaign and its model requests. The full campaign has not
been completed; a development rehearsal is not a benchmark campaign.

## Measured development results

A nine-case FP32 pilot on a **14-core M3 Pro GPU** promoted **2 of 9 signatures**:
medium bias/GELU measured **1.547–1.553× versus Inductor fullgraph**, and large
bias/GELU **1.093–1.097×**. Seven cases did not win. Each winner passed 22 private
fixtures and two independent timing sessions. These are **assumed-domain**
public-call measurements; they do not describe default checked-mode performance.

[All outcomes, source, raw samples, confidence intervals, and reproduction](results/2026-11-02-fp32-pilot/README.md)
are preserved. A separate unchanged matmul-plus-activation example measured
1.153× whole-workflow gain in one session. This is not whole-model confirmation.
The full 108-job comparison and feedback ablation remain unexecuted.

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
environments use the original function. Arbitrary models, general graph
replacement, backward generation, and cross-chip portability are future work.

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
- [Demo script](docs/demo.md)

PyTorch is pinned to 2.15.0. GPU checks skip explicitly when MPS is unavailable.
Use `requirements-dev.lock` for the tested dependency snapshot. Event-specific
model eligibility and a full evaluation campaign must be verified before a
hackathon submission makes broader claims.
