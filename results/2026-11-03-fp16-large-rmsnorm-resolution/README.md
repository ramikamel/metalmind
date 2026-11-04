# Large FP16 RMSNorm resolved as no win

The saved kernel passed 22 fresh private correctness cases. It also
completed two stable timing sessions against every required PyTorch baseline.
It made **zero Meta requests**. On this 14-core M3 Pro and exact FP16
shape (512, 4096), under explicit caller-guaranteed input bounds, the result
was **no win**. MetalMind kept the PyTorch path.

| Verified baseline | Session 1 speed ratio | Session 2 speed ratio |
| --- | ---: | ---: |
| Eager PyTorch | 17.312× | 18.081× |
| PyTorch functional RMSNorm | 5.678× | 5.544× |
| PyTorch Inductor fullgraph | **0.997×** | **0.990×** |

The ratio divides baseline latency by candidate latency. The paired 95%
intervals against Inductor were **0.994–1.019×** and **0.979–1.005×**.
Both include parity. The results support a large gain over eager execution
and essentially equal performance with the fastest verified baseline in
these sessions. The promotion rule requires at least 1.05× and a lower
interval bound above 1.00× in each session.

## What made the timing conclusive

The original calibration used the larger of the two repetition counts.
This brought the faster callable up to the block target, while the much
slower eager callable ran for far longer. An earlier complete first
session spent about 165 seconds in timed blocks across its three baselines.
The original execution cap was 120 seconds. Increasing the cap let the
benchmark finish individual windows, but baseline drift still invalidated
the second sessions in both previous rechecks.

This separate experiment used the smaller of the calibrated repetition counts.
Both arms still used the same count within a pair. The slower arm set the
block duration, and the faster arm could finish its block sooner. Each baseline
still received 30 balanced pairs, a control measurement, and the same 10% drift
limit. Both sessions passed on their first attempts. The timed blocks took
14.8 and 20.1 seconds across all three baselines. The largest observed control
drift was 4.24%. These totals exclude setup and the independent-session cooldown.

This diagnostic used the same saved source, shape, dtype and performance gate.
Its block calibration differs from the original pilot. Earlier faster-looking
Inductor ratios came from separate sessions whose second confirmations were
invalid. This result closes the follow-up decision while the archived pilot and
earlier rechecks preserve their original outcomes.

[Protocol](protocol.json) · [Result](summary.json) · [First session](session-1-attempt-1.json)
· [Second session](session-2-attempt-1.json) · [File hashes](sha256.json)

To reproduce, copy [benchmark.py](benchmark.py) into a fresh ignored directory
under `runs/` and run the copy from the repository root. It uses the archived
candidate, writes new private fixtures and timing records beside the copy,
and needs a matching Apple Silicon environment.
