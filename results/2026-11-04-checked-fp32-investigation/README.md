# Checked FP32 performance investigation

The original checked-mode campaign promoted zero kernels in 108 jobs.
A later same-kernel probe showed that input-domain checks dominated the
medium FP32 bias/GELU public call. We tested whether fewer scans and fewer
GPU commands would close that gap on the same 14-core Apple M3 Pro.

The two scan prototypes both preserve the existing numerical policy. One uses
a GPU flag initialized to zero, and the other writes separate SIMD group
flags without initialization. Both scan the stored IEEE bits and check the
flag before running the kernel. Invalid inputs still call PyTorch. Each
prototype passed 292 parity cases across four previously promoted FP32
kernels, including signed zeros, subnormals, infinities, NaNs, and values on
either side of the bounds.

| Workload | Old checked ÷ one flag | Old checked ÷ group flags | Fastest PyTorch ÷ prototype, one flag / group flags |
| --- | ---: | ---: | ---: |
| bias_gelu:medium | 2.30× | 2.27× | 0.128× / 0.127× |
| bias_gelu:large | 2.84× | 2.77× | 0.847× / 0.846× |
| softmax:medium | 1.44× | 1.45× | 0.109× / 0.111× |
| softmax:large | 1.74× | 1.67× | 0.169× / 0.166× |

A third prototype fused the bit checks into large bias/GELU so the kernel
could use the values it already loaded. It passed 10
public fixtures and 60 boundary cases. It ran
3.07× and 3.00× as fast as the old checked wrapper, and about 1.19× as fast
as eager PyTorch. It reached 0.376× and 0.370× against Inductor in those
measurements. These were diagnostics with ten pairs each. All prototypes
kept PyTorch as the performance winner against the strongest baseline.

## Guard-only latency

The single-flag scan by itself took longer than a full PyTorch call in
8 of 9 shapes. The guard timing includes the GPU scan and the host
readback used to make the fallback decision. It excludes generated kernel
arithmetic. Timings below are medians in milliseconds.

| Workload | Fastest verified PyTorch baseline | PyTorch (ms) | Guard only (ms) |
| --- | --- | ---: | ---: |
| bias_gelu:small | Eager PyTorch | 0.012 | 0.137 |
| bias_gelu:medium | Eager PyTorch | 0.053 | 0.138 |
| bias_gelu:large | Inductor | 0.303 | 0.192 |
| rmsnorm:small | Functional RMSNorm | 0.006 | 0.137 |
| rmsnorm:medium | Functional RMSNorm | 0.011 | 0.139 |
| rmsnorm:large | Functional RMSNorm | 0.127 | 0.188 |
| softmax:small | Eager PyTorch | 0.007 | 0.136 |
| softmax:medium | Eager PyTorch | 0.018 | 0.136 |
| softmax:large | Eager PyTorch | 0.030 | 0.146 |

These probes measure three approaches on the existing checked boundary.
A new prompt can change generated kernel work, while the cost of returning
the guard result to the host is outside the generated kernel. The historical
108-job result remains the record for that campaign.

[One-flag data](measurements.json) · [Group-flags data](flags_measurements.json) ·
[Fused GELU data](fused_gelu_measurements.json) · [Guard-only data](floor_measurements.json) ·
[Original campaign](../2026-11-03-checked-fp32-full/README.md)
