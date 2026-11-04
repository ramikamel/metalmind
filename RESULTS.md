# MetalMind benchmark results

On a 14-core Apple M3 Pro GPU, MetalMind confirmed faster kernels for **9 of 9
FP16 signatures** and **4 of 9 FP32 signatures** with caller-guaranteed input
bounds. The separate default checked-mode FP32 campaign promoted **0 of 108
jobs**.

## Function benchmarks

Speedup is PyTorch baseline latency divided by MetalMind latency. Above 1.00×
favors MetalMind. The two values for each promoted kernel are its independent
confirmation sessions against the fastest verified PyTorch baseline. Each
promoted kernel also passed 22 private correctness cases. Search scores for
the five FP32 fallbacks describe tested candidates. The deployed path in those
cases was PyTorch.

### FP16 with caller-guaranteed input bounds

| Workload | Shape | PyTorch baseline | Speedup | Outcome |
| --- | --- | --- | ---: | --- |
| Bias/GELU, small | 64 × 257 | Eager PyTorch | 1.660× / 1.689× | Promoted |
| Bias/GELU, medium | 256 × 1024 | Inductor | 1.657× / 1.651× | Promoted |
| Bias/GELU, large | 512 × 4096 | Inductor | 1.240× / 1.242× | Promoted |
| RMSNorm, small | 64 × 257 | Functional RMSNorm | 1.136× / 1.135× | Promoted |
| RMSNorm, medium | 256 × 1024 | Inductor | 1.541× / 1.530× | Promoted |
| RMSNorm, large | 512 × 4096 | Inductor | 1.252× / 1.268× | Promoted |
| Softmax, small | 64 × 257 | Eager PyTorch | 1.113× / 1.119× | Promoted |
| Softmax, medium | 256 × 1024 | Eager PyTorch | 3.098× / 3.183× | Promoted |
| Softmax, large | 128 × 4096 | Inductor | 1.428× / 1.420× | Promoted |

### FP32 with caller-guaranteed input bounds

| Workload | Shape | PyTorch baseline | Speedup | Outcome |
| --- | --- | --- | ---: | --- |
| Bias/GELU, small | 64 × 257 | Eager PyTorch | 0.923× in search | Kept PyTorch |
| Bias/GELU, medium | 256 × 1024 | Inductor | 1.589× / 1.568× | Promoted |
| Bias/GELU, large | 512 × 4096 | Inductor | 1.080× / 1.084× | Promoted |
| RMSNorm, small | 64 × 257 | Functional RMSNorm | 0.381× in search | Kept PyTorch |
| RMSNorm, medium | 256 × 1024 | Functional RMSNorm | 0.618× in search | Kept PyTorch |
| RMSNorm, large | 512 × 4096 | Functional RMSNorm | 0.942× in search | Kept PyTorch |
| Softmax, small | 64 × 257 | Eager PyTorch | 0.495× in search | Kept PyTorch |
| Softmax, medium | 256 × 1024 | Eager PyTorch | 1.216× / 1.228× | Promoted |
| Softmax, large | 128 × 4096 | Eager PyTorch | 1.173× / 1.181× | Promoted |

## Whole-workflow benchmarks

| Workflow | PyTorch baseline | Speedup |
| --- | --- | ---: |
| FP16 attention block | Eager split attention | 1.173× / 1.173× |
| FP16 attention block | Built-in attention | 1.099× / 1.091× |
| FP16 attention block | Inductor split attention | 1.058× / 1.067× |
| FP32 matmul and bias/GELU | Eager PyTorch | 1.153× in one session |
| FP32 residual module rewrite | Eager PyTorch | 1.123× / 1.112× |
| FP32 residual module rewrite | Inductor | 0.426× / 0.433× |

The FP16 attention benchmark used an earlier medium softmax artifact in a
fixed, manually integrated block. Timing included both matrix multiplications.
The FP32 residual rewrite lost to compiled PyTorch, so MetalMind kept the
original module. The FP32 matmul and bias/GELU measurement
used one session against eager PyTorch.

## Default checked mode

All 108 checked-mode FP32 jobs kept PyTorch. The campaign recorded 91 no-win,
4 budget-exhausted, and 13 failed jobs. In a separate same-kernel diagnostic for
medium FP32 bias/GELU, the checked call took about 0.41 ms versus 0.024 ms with
caller-guaranteed bounds. [Checked campaign evidence](results/2026-11-03-checked-fp32-full/README.md) ·
[Same-kernel diagnostic](results/2026-11-03-hackathon-extension/README.md#what-checked-mode-costs-on-the-same-kernel)

## Verification and source data

The release passed 195 local tests, including real Metal checks, and
[GitHub CPU CI](https://github.com/ramikamel/metalmind/actions/runs/41843792818).
[FP32 confirmations](results/2026-11-03-hackathon-extension/fp32-revalidation/README.md) ·
[FP16 measurements](results/2026-11-03-assumed-fp16-pilot/README.md) ·
[FP16 artifact checks and workflow evidence](results/2026-11-03-hackathon-extension/README.md) ·
[FP32 module evidence](results/2026-11-02-module-baselines/README.md) ·
[Prompt tuning and strongest FP16 softmax evidence](results/2026-11-04-muse-prompt-ab/README.md)
