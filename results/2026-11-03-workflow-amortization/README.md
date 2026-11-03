# Reuse cost for the medium bias/GELU artifact

The saved FP32 matmul-plus-activation workflow on a 14-core M3 Pro ran
**1.153× as fast** with the guarded bias/GELU artifact as with the original
eager activation. The paired bootstrap interval was **1.142–1.160×** over
30 pairs in one session. The median within-pair saving was **29.1 µs per
workflow call**. This used the artifact's explicit assumed-domain policy.

The original search recorded **69.59 seconds**. Adding the configured
**60-second** cooldown between the two successful function confirmations gives
**129.59 seconds** of accounted setup cost. At the observed median saving,
it takes **4,454,974 workflow calls** to offset that cost. This is a partial
estimate for generating and confirming the artifact once. Private verification,
confirmation measurement time, export, and first-load compilation add further
time. Their separate durations were unavailable in the archived data. The
estimate assumes the observed per-call saving persists and uses an eager
whole-workflow baseline.

## Reuse of an existing artifact

Five fresh Python processes loaded the same promoted artifact offline on the
same Mac. The median load was **199.63 ms**, including evidence checks,
environment detection and the Metal compile call. The median Metal compile
call itself took **1.12 ms**. At the earlier median saving of 29.1 µs, the
measured load offsets after about **6,863 workflow calls**. This is a separate
estimate for an existing artifact. Python startup and imports occur outside
the timer, and the system shader-cache state was uncontrolled.

[Raw first-load samples](first-load.jsonl) are separate from the historical
search estimate in `report.json`.

From the checkout, regenerate [report.json](report.json) using the original
[FP32 pilot evidence](../2026-11-02-fp32-pilot/README.md):

```sh
pilot=results/2026-11-02-fp32-pilot
artifact=db066f6dfe2e74cefc0c335852229dd2ed6a9058518279eabb7b3a28342caa87
python -m scripts.report_workflow_amortization \
  --run "$pilot/jobs/bias_gelu-medium" \
  --artifact "$pilot/artifacts/$artifact" \
  --workflow "$pilot/pilot-integrated-block.json"
```

The report verifies the promoted artifact, matches its identity, shape and
domain to the saved workflow, and derives the estimate directly from the
archived timing pairs. Regeneration requires neither GPU access nor a model key.

Run this once per fresh process to reproduce the first-load sample on the
matching Mac:

```sh
python -m scripts.probe_artifact_first_load --artifact "$pilot/artifacts/$artifact"
```
