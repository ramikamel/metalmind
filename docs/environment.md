# Verified development environment

Updated November 2, 2026. These are observed capabilities on this machine,
not minimum requirements or a cross-device performance claim.

| Component | Observed |
| --- | --- |
| GPU | Apple M3 Pro, **14 GPU cores** |
| Architecture | arm64 |
| macOS | 27.2 |
| Python | 3.14.6 |
| PyTorch | 2.15.0 |
| MPS | Built and available |
| Direct shaders | `torch.mps.compile_shader` works |
| Pipeline limit probe | 1,024 threads per threadgroup |
| Framework compilation | Inductor fullgraph usable for the recorded GELU signature |
| Generation model | `muse-wonder-1.0`, Meta Responses API |

Use `requirements-dev.lock` for the tested macOS dependency snapshot. Python
3.14.6 was already installed and passed capability checks. The original design's
Python 3.12 suggestion is not a measured GPU environment. CPU CI uses Python 3.12
and deliberately does not claim Metal performance validation.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.lock
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/metalmind doctor --json --smoke
.venv/bin/python -m pytest -q
```

The runtime suite covers real compilation and launch, mixed uint32/FP32 ABI
arguments, irregular counts, runtime tensor parameter changes, typed compiler
errors, signature/alias/autograd guards, input integrity, numerical rejection,
FP16 intermediates, and synchronized timing. Run GPU tests on an otherwise idle
GPU. Drift invalidates measurements; an unstable timing test may skip and is
never evidence of a win.

Model access and generation were explicitly exercised using the ignored local
`.env`; ordinary tests use fake/replay providers. No key is included in evidence.
The key is not needed for report generation or offline artifact reuse.

The package wheel was installed with its dependencies in a fresh temporary
virtual environment outside this checkout, without `MODEL_API_KEY` or `PYTHONPATH`.
The import came from that environment's site-packages, the console entry point
reported the expected runtime, and the 108-job campaign preview completed without
a model call. The check record is retained with release evidence.

Artifact fingerprints include the runtime report, relevant MPS flags, GPU model,
GPU core count, Metal-family capability, and recommended working set. Different
hardware/software disables reuse until explicit revalidation. The strengthened
fingerprint was exercised by a fresh successful private/performance revalidation;
see [evidence](../results/2026-11-02-revalidation/README.md).

`timed_events` in diagnostics means API presence. Real timing and shader health
are separate tests. Shader Validation and CPU fallback are rejected during
performance measurement. Controller evaluation deadlines are 120 seconds after
calibration, under the absolute worker and whole-job budgets.


Module continuation: the complete local suite now passes **113 tests**, no skips.
The updated wheel also passed isolated installation, real shader health,
`analyze-module` extraction and `optimize-module` no-artifact fallback outside
the checkout with no credentials. Exact module release identities and all three
development outcomes are in [module evidence](../results/2026-11-02-module-optimizer/README.md).
