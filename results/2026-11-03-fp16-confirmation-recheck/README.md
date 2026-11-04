# FP16 confirmation rechecks

The original [nine-job FP16 pilot](../2026-11-03-assumed-fp16-pilot/README.md)
retains its six promotions and three inconclusive outcomes. This separate
follow-up rechecked the same selected kernels. It made **zero Meta requests**
and generated no replacement source.

| Signature | Original confirmation problem | Follow-up outcome |
| --- | --- | --- |
| Small bias/GELU (64, 257) | Second session rejected for baseline drift | **Promoted on pass 2** with 1.660× and 1.689× against eager PyTorch, the fastest verified baseline |
| Small RMSNorm (64, 257) | First session rejected for baseline drift | **Promoted on pass 1** with 1.136× and 1.135× against PyTorch functional RMSNorm, the fastest verified baseline |
| Large RMSNorm (512, 4096) | First session reached the worker deadline | **Inconclusive** after two attempts. First sessions measured 1.247× and 1.250× against Inductor. Both second sessions were rejected for baseline drift. |

All original candidates passed their private numerical gates. Baseline drift
means the baseline's median latency during the paired run differed by more
than 10% from its control measurement. The harness discarded each affected
window. In the first valid large-RMSNorm recheck, the timed blocks alone
took about 165 seconds across three baselines, exceeding the original
120-second execution cap even before accounting for setup.

The first pass used fresh private inputs, two independent confirmation
sessions, a 300-second execution cap, and the original 25 warmups and
60-second cooldown. Only still-inconclusive cases entered pass 2, with 100
warmups, a 120-second cooldown, and a 450-second execution cap. Both passes
retained the same FP16 assumed-domain boundary, drift limit, required
baselines, pair count, numerical tolerances and promotion thresholds. The
original search measurements were copied as baseline requirements. Fresh
confirmation measurements determined the new outcomes.

Both promoted artifacts passed the [offline reuse audit](reuse-audit.json)
on new values. Shape, FP32 input and gradient fallback each used PyTorch.
A [subsequent instrumented experiment](../2026-11-03-fp16-large-rmsnorm-resolution/README.md)
resolved large RMSNorm as no win against Inductor, with two stable sessions.

The rechecks bring the number of FP16 signatures with a confirmed win to
**eight of nine across the original pilot and this follow-up**.

[Attempt summary](summary.json) · [File hashes](sha256.json)

The archived procedure files record the exact commands' Python logic.
To repeat a pass, copy its procedure as `recheck.py` into a fresh ignored
directory under `runs/` and invoke it from the repository root. Pass 2
takes the new pass 1 summary path as its argument. The script writes new
evidence beside itself.
