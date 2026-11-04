# MetalMind judging rehearsal

## The demo moment

> Muse Wonder wrote an Apple GPU kernel. MetalMind made it earn the right to run.

Open the [offline evidence showcase](../demo/evidence-showcase.html) in a
browser. Start on the **2.54×** medium FP16 softmax result, which is the lower
of two measured speedups against the fastest verified baseline in each
session. Say “for a fixed shape on this M3 Pro, with caller-guaranteed input
bounds.” Scroll to **Search trace**, expand the real Metal source, and show
the other proposal that produced a non-finite output and was rejected.
These are saved model responses and benchmark results.

Then run the command below on the matching Mac. It actually loads the winning
artifact, runs a fresh FP16 input on the GPU, and counts kernel dispatches.
The matching input dispatches once. Changed shape, FP32 inputs and gradient
mode use PyTorch. Point to `observed_kernel_dispatches` and the fallback flags.
This check needs no Meta requests.

```sh
cd ~/github/metalmind
artifact=$(.venv/bin/python -c 'import json; from pathlib import Path; r=Path("results/2026-11-03-assumed-fp16-pilot"); rows=json.loads((r/"summary.json").read_text())["rows"]; print(r/"raw"/next(j["artifact_path"] for j in rows if j["family"]=="softmax" and j["case"]=="medium"))')
.venv/bin/python -m scripts.audit_fp16_reuse "$artifact"
```

Close with the large RMSNorm no-win card. Eager PyTorch was 17–18× slower,
while Inductor roughly tied the candidate. The agent kept PyTorch. This
shows the same decision rule protecting a developer from a misleading
benchmark. The [recorded walkthrough](../demo/video/metalmind-evidence-walkthrough.mp4)
can carry the presentation if live timing or hardware is unavailable.

## How the evidence maps to judging

| Official category | What to show |
| --- | --- |
| Impact · 30% | A developer on Apple Silicon can find reusable kernels for repeated fixed-shape inference. Show the fresh use of an existing artifact and the measured matmul-plus-activation workflow. |
| Use of Meta AI Tooling · 25% | Show Muse Wonder's actual Metal and how the agent uses compiler, correctness and benchmark feedback to guide later attempts. The live generation ledger makes Meta's role visible. |
| Technical Depth & Execution · 20% | Show the numerical rejection, 22 private cases, two confirmation sessions, verified compiler comparison, and guarded fallback. |
| Demo · 25% | Show one archived win, one fresh GPU dispatch with fallback, and one measured no-win. Narrate what the viewer is seeing and keep the mode labels visible. |

The public rubric scores a working and impressive demo, live reliability, and
video quality within the Demo category. The Indie Developer track and Push the
Frontier theme fit the project. The event lists Muse Wonder, and the published
deadline is November 4, 2026 at 9 p.m. Eastern, 8 p.m. Chicago time. The submit
link requires sign-in. The exact form and media limits are visible after sign-in.
[Official rules](https://cerebralvalley.ai/e/meta-ai-developer-hackathon/waiver) · [Event page](https://cerebralvalley.ai/e/meta-ai-developer-hackathon) · [Submission portal](https://cerebralvalley.ai/e/meta-ai-developer-hackathon/hackathon/submit)
