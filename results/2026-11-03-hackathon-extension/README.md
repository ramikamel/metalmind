# Deeper Muse Wonder search and Metal measurement audit

Measured November 3–4, 2026 on an Apple M3 Pro with a 14-core GPU, Python 3.14.6,
PyTorch 2.15.0 and macOS 27.2. The search selected four cases that had never
promoted in the earlier studies. All paid searches used the assumed-domain
contract. This targeted follow-up has its own denominator and leaves the original
pilots and 108-job checked campaign as historical records.

## What Muse Wonder found

The live run completed 24 requests to `muse-wonder-1.0`, six for each case.
The response ledger records 63,494 input tokens and 103,399 output tokens,
including 75,747 reasoning tokens. All 24 requests yielded candidates for
execution. The per-case budget was six generations at medium reasoning effort,
with up to 8,192 output tokens per request. [Protocol](search/protocol.json) ·
[Search outcomes](search/summary.json) · [Search procedure](search/procedure.py).

| Assumed-domain case | Strongest verified baseline | Outcome | Baseline ÷ Metal |
| --- | --- | --- | --- |
| FP32 medium softmax `(256, 1024)` | Eager PyTorch | Promoted in both original search confirmations | 1.212× and 1.206× |
| FP32 large softmax `(128, 4096)` | Eager PyTorch | Promoted in both original search confirmations | 1.181× and 1.188× |
| FP32 medium RMSNorm `(256, 1024)` | PyTorch functional RMSNorm | No win | Best search candidate 0.618× |
| FP16 large RMSNorm `(512, 4096)` | Inductor fullgraph | Promoted on the second short-block recheck | 1.252× and 1.268× |

The first FP16 large RMSNorm study had an earlier candidate at 0.997× and
0.990× versus Inductor. The follow-up gave the model that outcome and the old
source. The new incumbent processes two rows per threadgroup, caches weight
vectors in registers and uses SIMD reductions. It reached 1.274× in its first
legacy confirmation. Baseline drift invalidated the second legacy session and
the second session of the first short-block recheck. Each recheck started a fresh
private gate. The decisive attempt passed 22 private cases and measured 1.252×
(95% interval 1.245–1.260×) and 1.268× (1.256–1.283×) versus Inductor. It also beat
functional RMSNorm by 6.314× and 6.364×. No model calls were made during either
recheck. All windows and attempts are retained. [Earlier no-win](../2026-11-03-fp16-large-rmsnorm-resolution/README.md) ·
[Short-block attempts](short-block-recheck/summary.json) · [Fresh reuse checks](reuse.json).

Across the original nine FP16 signatures and their follow-ups, there is now at
least one promoted assumed-domain artifact per signature. The original FP16 pilot
promoted six. The two intervening confirmations brought the count to eight, and
this follow-up added the ninth. The matching FP16 RMSNorm and softmax artifacts
both dispatched once on fresh inputs. Changing shape, using FP32 inputs or
asking for gradients used PyTorch. [Original FP16 pilot](../2026-11-03-assumed-fp16-pilot/README.md) ·
[Intervening confirmations](../2026-11-03-fp16-confirmation-recheck/README.md).

## FP32 wins revalidated with shorter blocks

The original FP32 pilot promoted two of nine signatures. The deeper search
added medium and large softmax, bringing the total across studies to four
of the same nine signatures. All four reproduced with the corrected timer
and fresh private cases. Speedups below use each case's fastest verified baseline.

| Exact FP32 artifact | Baseline | Fresh confirmation sessions |
| --- | --- | --- |
| Medium bias/GELU | Inductor fullgraph | 1.589× and 1.568× |
| Large bias/GELU | Inductor fullgraph | 1.080× and 1.084× |
| Medium softmax | Eager PyTorch | 1.216× and 1.228× |
| Large softmax | Eager PyTorch | 1.173× and 1.181× |

The FP32 medium softmax required a third formal attempt with a longer warmup
because the first two attempts encountered drift. All attempts are retained.
[Exact-source FP32 results, including every attempt](fp32-revalidation/README.md).

## What checked mode costs on the same kernel

The audit ran the exact archived medium FP32 bias/GELU kernel with two public
wrappers. Both matched PyTorch on ten in-domain fixtures. The checked wrapper
also called PyTorch for a verified out-of-domain input. The content scan caused
roughly 17.36× and 17.43× longer calls than the assumed-domain wrapper, around
0.41 ms versus 0.024 ms in the direct comparisons. Both wrappers included
signature guards, allocation, dispatch and completion.

| Baseline | Checked wrapper, sessions 1 and 2 | Assumed-domain wrapper, sessions 1 and 2 |
| --- | --- | --- |
| Eager PyTorch | 0.128× and 0.128× | 2.331× and 2.352× |
| Inductor fullgraph | 0.059× and 0.057× | 1.007× and 1.032× |

The same-kernel probe isolates the mode overhead for this one signature. It also
found that the earlier 1.55× GELU advantage against Inductor did not repeat in
these diagnostic sessions, although the separately run formal revalidation
measured 1.589× and 1.568×. The same kernel can have different ratios under a
mixed-mode diagnostic and a dedicated formal run. The available evidence does
not isolate the cause of that difference. The historical 108-job checked campaign promoted zero kernels
and used the older timer and value guard. [Raw workflow measurements](workflow-measurements.json) ·
[Checked FP32 campaign](../2026-11-03-checked-fp32-full/README.md).

## A complete FP16 attention block

A manually integrated `(Q @ K.T * 0.125) → softmax → @ V` block reused the
medium FP16 softmax artifact for fixed `Q=(256,64)`, `K=V=(1024,64)`. Only softmax
changed. All three PyTorch baselines and the candidate passed the complete output
check across eight in-domain fixtures per session. The timing covered both
matrix multiplications and the softmax, with 30 paired measurements per session.
The block used assumed-domain FP16 and synthetic dyadic inputs whose score
magnitudes satisfy the artifact contract. This is a single fixed attention block.

| Complete PyTorch baseline | Session 1 (95% interval) | Session 2 (95% interval) |
| --- | --- | --- |
| Eager split attention | 1.173× (1.160–1.195×) | 1.173× (1.166–1.191×) |
| Built-in scaled dot-product attention | 1.099× (1.086–1.124×) | 1.091× (1.079–1.114×) |
| Inductor fullgraph split attention | 1.058× (1.047–1.071×) | 1.067× (1.052–1.080×) |

Inductor was the fastest verified whole-block baseline. Its paired results give
about 1.06× whole-block speedup, while the isolated artifact's published softmax
result was 2.54× against its strongest operation baseline. The audit had no API
calls and used the current short-block timer. [Raw workflow measurements](workflow-measurements.json).

## Method and reproduction

The paid search used the source at `be110a5` (package SHA-256
`20a56fb61aabb27803ae9bd3d6cdb5dc3c3a3857930be855efee6920d4fec481`). It kept the
legacy larger-repetition timing throughout. The zero-request audit and FP16
rechecks used the source at `c8313df` (package SHA-256
`dab50186f60f68d4f1693835d406cdf59504c0caff7c687411807c709d8f50df`). Short-block
calibration lets the slower arm choose the common repetition count. Two
confirmation sessions, a 60-second cooldown, the 10% drift rule, 30 paired samples,
and the existing promotion threshold stayed in force. The new value guard uses
IEEE magnitude bits to distinguish stored FP32 subnormals from signed zero.
[Shader health](doctor.json) · [Test run](tests.txt) · [Benchmarking contract](../../docs/benchmarking.md).

From the repository root on the matching hardware, the following create fresh
zero-request evidence in a temporary directory. The recheck selects the exact
legacy incumbent when the original timing was inconclusive after a qualifying
search score. [Archived sources and ledgers](search/) include the unsuccessful
candidates and confirmation results.

```sh
archive=results/2026-11-03-hackathon-extension
output=$(mktemp -d)
python -m scripts.recheck_hackathon_incumbents "$archive/search" "$output/recheck"
python -m scripts.probe_hackathon_workflows --output "$output/workflow.json"
(cd "$archive" && shasum -a 256 -c SHA256SUMS)
```
