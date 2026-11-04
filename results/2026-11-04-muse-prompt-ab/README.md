# Muse Wonder prompt comparison

This exploratory A/B test compared the existing MetalMind prompt with an
experimental prompt that included operation-specific Metal guidance and a
shorter summary of previous timings. The tuned prompt told Muse which
verified PyTorch baseline was hardest to beat. Both versions retained
the same Metal ABI, numerical contract and promotion gate.

The experiment used **16 live Muse Wonder requests** on a
14-core Apple M3 Pro. Both arms had a two-request budget per signature,
medium reasoning, the same seeds for timing, and assumed-domain inputs.
The prompt order alternated across workloads.

| Workload | Current prompt | Tuned prompt |
| --- | --- | --- |
| FP32 small bias/GELU | 0.963× search, kept PyTorch | 0.882× search, kept PyTorch |
| FP16 medium softmax | 2.735× search, promoted at 2.713× / 2.741× | 2.765× search, failed |
| FP32 medium RMSNorm | 0.543× search, kept PyTorch | — search, budget exhausted |
| FP16 large RMSNorm | 1.220× search, promoted at 1.202× / 1.214× | 1.235× search, promoted at 1.226× / 1.228× |

Search scores are the best valid candidate ratios against each candidate's
fastest verified baseline. Only promoted entries passed the private gate
and both confirmation sessions. Candidate search timing and promotion
are distinct measurements.

| Aggregate | Current | Tuned |
| --- | ---: | ---: |
| Requests | 8 | 8 |
| Candidates evaluated | 8 | 8 |
| Candidates with recorded correctness pass | 6 | 5 |
| Valid timed candidates | 6 | 5 |
| Promotions | 2 | 1 |
| Input tokens | 8182 | 6316 |
| Output tokens | 29141 | 28662 |

The current prompt produced two promotions and the tuned prompt produced one.
Both promoted large FP16 RMSNorm, where the tuned kernel measured slightly
faster in the two confirmation sessions. The current prompt remains the
default for MetalMind.

This test had one short search per prompt and signature. It measures the
combined prompt changes on these cases. Variation from model sampling and
timing may account for some differences.

[Protocol](protocol.json) · [Summary](summary.json) · [Procedure](procedure.py)

Run the procedure from the repository root on the matching Mac with the
project environment activated and a configured Meta API key. This makes
live requests and writes fresh evidence to the supplied directory.

```sh
python results/2026-11-04-muse-prompt-ab/procedure.py "$(mktemp -d)"
```

## Focused softmax repair

The tuned softmax candidate reached 2.765× in search and failed private
verification because its output varied across identical inputs. Its SIMD groups
could overwrite the shared maximum with a partial sum while other groups still
needed that value. The zero-request diagnostic replayed the same 22 fixtures
three times for the original source and for a manual control with an added
barrier. The original passed 0 of 3 evaluations and reproduced
nondeterminism in 3 of 3. The manual control passed 3 of 3.

A separate focused prompt then sent Muse the failing source and a specific
synchronization hint. It used 2 additional live requests.
Muse first separated the maximum and sum scratch buffers, then proposed a
kernel that assigned one SIMD group to each row and kept intermediate
values in registers. The latter passed 22 fresh private cases and measured
**3.098× and 3.183×** against eager PyTorch, the fastest verified baseline
in both sessions. A fresh offline check observed one Metal dispatch and
confirmed fallback for shape, dtype, and gradient changes.

This follow-up tested repair of one known failure. The A/B comparison above
used fresh searches with no supplied incumbent.
[Repair procedure](repair_procedure.py) · [Race diagnostic](focused-repair/diagnostic.json) ·
[Focused result](focused-repair/summary.json) · [Offline reuse](focused-repair/reuse.json)

To reproduce the focused follow-up from the archived failing source, run

```sh
experiment=results/2026-11-04-muse-prompt-ab
python "$experiment/repair_procedure.py" "$experiment" "$(mktemp -d)/repair"
```
