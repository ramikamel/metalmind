# Record the MetalMind demo

Aim for **2 minutes 30 seconds**. Show one FP16 softmax kernel from the saved
failure and repair through a fresh GPU dispatch. The commands used on camera
make zero Muse requests. The generation and timing evidence on screen comes
from the saved experiments.

## Set up the screen

- Open the [README flowchart](../README.md#system-flow), the [softmax repair report](../results/2026-11-04-muse-prompt-ab/README.md#focused-softmax-repair), and [RESULTS.md](../RESULTS.md) in that order. Open the README on GitHub and fit the whole flowchart in the recording frame.
- In the repair report, pre-open the [original rejection](../results/2026-11-04-muse-prompt-ab/runs/float16-softmax-medium/tuned/jobs/684bde683ba747d7a0fd1c6e29fcd3cd/private-verification.json) and the [winning Metal source](../results/2026-11-04-muse-prompt-ab/focused-repair/artifacts/3c43a8a180ea90c420d51020d22d580ecbd0440f2b978423440489beeba03fd4/kernel.metal). Scroll the kernel to the SIMD reductions so they are visible when you switch tabs.
- Use one browser window and one terminal window. Increase text size enough to read the labels in a playback window. Close unrelated tabs, hide notifications, and keep the environment file and API key outside the recording.
- Confirm the submission portal's current video fields before choosing an upload format or cutting the video to a different length. [Submission portal](https://cerebralvalley.ai/e/meta-ai-developer-hackathon/hackathon/submit)

Before recording, run these commands from the repository root in the terminal
you will use on camera. The path lookup selects the exact softmax artifact
shown in the repair report.

```sh
cd ~/github/metalmind
.venv/bin/metalmind doctor --smoke --json

focus=results/2026-11-04-muse-prompt-ab/focused-repair
artifact=$(.venv/bin/python -c 'import json, sys; from pathlib import Path; p = Path(sys.argv[1]); print(p / json.loads((p / "summary.json").read_text())["artifact_path"])' "$focus")
test -f "$artifact/kernel.metal"
```

Run the two on-camera commands below once as a rehearsal. Keep the same shell
open so the artifact variable is available when recording.

```sh
.venv/bin/metalmind analyze --function examples.fp16_workloads:row_softmax \
  --workload softmax --case medium --dtype float16
```

```sh
.venv/bin/python -m scripts.audit_fp16_reuse "$artifact"
```

The first command should show `supported: true`, `float16`, and a shape
array containing `256` and `1024`. It analyzes the importable example. The
audit loads the saved artifact for the registered version of the same softmax
calculation. It should show `new_values_match: true`, exactly one
`observed_kernel_dispatches`, all three fallback flags as `true`, and
`model_requests: 0`. The audit requires an environment compatible with the
saved artifact. If it fails the environment check, display the [saved audit](../results/2026-11-04-muse-prompt-ab/focused-repair/reuse.json)
and describe it as a recorded check. Keep the benchmark numbers tied to the
recorded M3 Pro measurements.

## Shot list and read-aloud script

| Time | Show on screen | Say |
| --- | --- | --- |
| **0:00–0:16** | GitHub README. Start on the project name, then reveal the system flowchart. | “I built MetalMind to optimize PyTorch functions on Apple Silicon. Muse Wonder proposes Metal kernels. MetalMind compiles them, checks the outputs against PyTorch, and keeps a kernel only if it also beats the verified baselines.” |
| **0:16–0:34** | Show the one-line [PyTorch softmax function](../examples/fp16_workloads.py), then run the `analyze` command. Point to `[256, 1024]` and `float16` in the output. | “This function computes row softmax with FP32 intermediates and returns FP16. MetalMind recognizes the pattern and records the exact shape and dtype. That specification tells Muse what the kernel has to implement.” |
| **0:34–0:58** | Open the original rejection JSON. Highlight `nondeterministic output`. Switch to **Focused softmax repair** and point to the three failed repeats and the barrier check. | “In a saved search, a fast candidate returned different values when the same input ran twice. MetalMind rejected it. I traced that to how SIMD groups reused shared memory, checked the diagnosis with a barrier, and gave Muse that specific feedback.” |
| **0:58–1:23** | Open the winning `kernel.metal`. Point to the SIMD reductions, then switch to the **Focused softmax repair** report and point to **3.098× / 3.183×**. | “Muse's winning revision uses one SIMD group per row and keeps intermediate values in registers. MetalMind checked 22 cases held out from search. The kernel then ran 3.10 and 3.18 times as fast as eager PyTorch, the fastest verified baseline in both sessions. These are saved measurements for this shape on a 14-core M3 Pro, with caller-guaranteed input bounds.” |
| **1:23–1:53** | Run the reuse audit in Terminal. Hold on the output and point to `new_values_match`, `observed_kernel_dispatches`, the three fallback flags, and `model_requests`. | “Now I'm loading that exact saved kernel and running fresh input values on this Mac. The output matches PyTorch and the audit observed one Metal dispatch. When I change the shape, change the dtype, or ask for gradients, MetalMind calls PyTorch. Loading and running this artifact takes zero model requests.” |
| **1:53–2:17** | In **RESULTS.md**, show **Whole-workflow benchmarks**, then scroll to **Default checked mode**. | “A fixed attention block using an earlier verified softmax artifact ran about 1.06 times as fast as Inductor, including both matrix multiplications. With the default input scan included, the separate checked FP32 campaign kept PyTorch in all 108 jobs. That scan took longer than a full PyTorch call in eight of nine measured FP32 shapes.” |
| **2:17–2:30** | Return to the README benchmark table. Hold on the repository name and the FP16 and FP32 counts. | “With guaranteed input bounds, we found verified kernels for nine of nine FP16 signatures and four of nine FP32 signatures. The source, the failed candidates, and the timings are in the repository.” |

If you need a shorter cut, keep the introduction, rejection, repair result,
and fresh audit. Drop the live `analyze` command and the attention comparison.
Keep the hardware, shape, and caller-guaranteed bounds in the narration.

## Record and review on macOS

1. Press **Shift–Command–5** and choose **Record Selected Portion**. Frame the browser and terminal area. Under **Options**, choose your microphone and a save location. Record a short test and play it back to check speech volume and text size. [Apple's screen recording instructions](https://support.apple.com/en-us/102618)
2. Start recording on the README. Follow the table in order. If a command takes longer than expected, pause your narration and trim that wait afterward. Hold each result long enough to read its label.
3. Stop the recording from the menu bar. Trim the start, end, and idle waits. Play the exported file once with sound and check that the browser text and terminal output are readable. Use the file format or link requested by the portal.

The repo also includes a [recorded FP32 walkthrough](../demo/video/metalmind-evidence-walkthrough.mp4)
and a [saved evidence showcase](../demo/evidence-showcase.html). They use
earlier artifacts and their corresponding results.
