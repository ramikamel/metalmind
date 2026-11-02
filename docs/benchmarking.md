# Measurement contract

The primary boundary is amortized synchronized public-function latency: guards,
configured value-domain checks, fresh output allocation, Python dispatch, and GPU
execution. Inputs are already resident on MPS. Compilation and model latency are
outside warm timing and must not be advertised as free.

Each candidate passes all public numerical and integrity fixtures before timing.
Framework baselines are also verified: original eager, functional RMSNorm where
applicable, and torch.compile/Inductor fullgraph when usable. Unsupported paths
are recorded, not omitted silently. CPU fallback is disallowed during comparison.

Use at least 25 warmups, calibrated common repetitions, balanced pair ordering,
ten search pairs, and thirty independent confirmation pairs. Bootstrap whole
pairs 10,000 times. Promotion requires at least 1.05x median public-function gain
and lower 95% bound above 1.0 in both confirmation sessions, separated by at least
60 seconds. A private numerical gate precedes confirmation. Drift beyond 10%
invalidates the window; no individual unfavorable samples are removed.

Checked mode scans input domains and includes scan/synchronization cost. Assumed
mode requires caller-guaranteed numerical bounds; its numbers are labeled
separately. The first tiny checked-mode kernel was slower despite being correct;
that run is development data and does not establish a general performance claim.

The frozen campaign preview has 108 jobs: nine FP32 signatures, three trials,
four strategies. It is not executed by preview. A full campaign, FP16 campaign,
cross-chip comparisons, and performance claims require their own saved evidence.


The development campaign aggregate assigns unsuccessful jobs a neutral factor
of 1.0 and uses each winner's worst confirmed baseline/session ratio. Its
geometric mean is a normalized research summary, not measured combined-workload
latency. Actual end-to-end effects are measured separately by
`examples.integrated_block`, whose baseline is unchanged eager matmul plus the
original activation. It is not a compiled whole-model comparison.

Campaign manifest version 2 freezes resolved workload shapes, reference hashes,
and correctness-policy values in addition to settings. The nine-case development
pilot began under the earlier configuration-only identity; its per-run workload
manifests remain authoritative, and its directory is preserved under that ID.
Future previews use version 2 and therefore have different IDs.
