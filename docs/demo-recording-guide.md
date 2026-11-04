# Record the MetalMind demo

**Target length: 2 minutes 30 seconds.** Follow the shots in order. The video
shows one FP16 softmax repair from a rejected candidate to a fresh offline
dispatch. The only two commands you run on camera make zero Muse requests.
All speedups come from saved experiments.

## Before you press Record

1. Open each GitHub link in the shots below in its own tab, in shot order.
   Put the browser and Terminal in the same recording area so you can switch
   between them. Set browser zoom so the full **System flow** diagram fits.
   Increase Terminal text size until the audit output is readable in playback.
2. In Terminal, run the setup below. Keep that Terminal tab open for the two
   on-camera commands. Rehearse both commands once, then clear Terminal
   with **Command–K** before each terminal shot. Keep the API key and
   `.env` file off screen.
3. Press **Shift–Command–5** → **Record Selected Portion**. In **Options**,
   select your microphone. Start on the GitHub flowchart with the full
   diagram visible.

```sh
cd ~/github/metalmind
.venv/bin/metalmind doctor --smoke --json
```

Check that the setup output says `"mps_available": true` and
`"shader_health_passed": true`. If the reuse audit still fails in rehearsal,
use the saved audit in shot 5. You can record the whole demo without live
generation.

## 1. Introduce MetalMind · 0:00–0:16

**Show:** [GitHub README → System flow](https://github.com/ramikamel/metalmind#system-flow).
Keep all three cards on screen. Move the pointer left to right over
**Propose**, **Test & select**, and **Reuse offline**. End on the bottom
**Run PyTorch** fallback bar.

**Say:** “I built MetalMind to optimize PyTorch functions on Apple Silicon.
Muse Wonder proposes Metal kernels. MetalMind compiles and checks them
against PyTorch, and keeps a kernel only if it also beats the verified
baselines. Otherwise, PyTorch still runs.”

## 2. Show the PyTorch input · 0:16–0:34

**Show:** [GitHub → `row_softmax`, lines 15–16](https://github.com/ramikamel/metalmind/blob/fff73af999b9c0591d58d46d45e5b112e98c6619/examples/fp16_workloads.py#L15-L16).
Point to `x.float()`, then `.to(x.dtype)` on the highlighted return line.
Switch to Terminal and paste:

```sh
(
  set -o pipefail
  .venv/bin/metalmind analyze --function examples.fp16_workloads:row_softmax \
    --workload softmax --case medium --dtype float16 |
    .venv/bin/python -c 'import json, sys; r = json.load(sys.stdin); print("supported:", json.dumps(r["supported"])); w = r.get("workload"); assert r["supported"] and w, r; i = w["inputs"][0]; print("shape:", i["shape"]); print("dtype:", i["dtype"]); print("accumulation_dtype:", w["accumulation_dtype"])'
)
```

**Point to these four lines:**

```text
supported: true
shape: [256, 1024]
dtype: float16
accumulation_dtype: float32
```

**Say:** “This is row softmax. It takes FP16 tensors and computes internally
in FP32. MetalMind recognizes the function and records the exact shape and
dtype so Muse knows what the kernel has to implement.”

## 3. Show why the first kernel failed · 0:34–0:58

**Show first:** [GitHub → original rejection, lines 3–12](https://github.com/ramikamel/metalmind/blob/fff73af999b9c0591d58d46d45e5b112e98c6619/results/2026-11-04-muse-prompt-ab/runs/float16-softmax-medium/tuned/jobs/684bde683ba747d7a0fd1c6e29fcd3cd/private-verification.json#L3-L12).
Point to `"status": "incorrect"`, then `"reason": "nondeterministic output"`.

**Show next:** [GitHub → Focused softmax repair](https://github.com/ramikamel/metalmind/blob/main/results/2026-11-04-muse-prompt-ab/README.md#focused-softmax-repair).
In the first paragraph under the heading, point to **“original passed 0 of 3
evaluations”** and **“manual control passed 3 of 3.”**

**Say:** “The first candidate looked fast, but it returned different values
for identical inputs. MetalMind rejected it. I traced the problem to GPU
threads reusing shared memory too early. A barrier fixed all three
diagnostic repeats, so I gave Muse that specific feedback.”

## 4. Show the repaired kernel and its speed · 0:58–1:23

**Show first:** [GitHub → winning Metal source, lines 18–34](https://github.com/ramikamel/metalmind/blob/fff73af999b9c0591d58d46d45e5b112e98c6619/results/2026-11-04-muse-prompt-ab/focused-repair/artifacts/3c43a8a180ea90c420d51020d22d580ecbd0440f2b978423440489beeba03fd4/kernel.metal#L18-L34).
Point to `simd_max(lm)` at line 18 and `simd_sum(ls)` at line 34. If the
browser only fits one line at a time, use the max line and keep moving.

**Show next:** Return to [GitHub → Focused softmax repair](https://github.com/ramikamel/metalmind/blob/main/results/2026-11-04-muse-prompt-ab/README.md#focused-softmax-repair).
Point to **“22 fresh private cases”**, then hold on **3.098× and 3.183×** and
the words **“against eager PyTorch”** in the next line.

**Say:** “Muse's revision assigns one group of GPU threads to each row.
MetalMind checked it on 22 cases held out from search. It ran 3.10 and 3.18
times as fast as eager PyTorch, the fastest verified baseline in both
sessions. Those are saved measurements for this shape on a 14-core M3 Pro,
with caller-guaranteed input bounds.”

## 5. Run the saved kernel offline · 1:23–1:53

**Show:** Switch to a clean Terminal and paste this command. It loads the
artifact from shot 4 for the registered version of the same softmax operation.

```sh
.venv/bin/python -m scripts.audit_fp16_reuse \
  results/2026-11-04-muse-prompt-ab/focused-repair/artifacts/3c43a8a180ea90c420d51020d22d580ecbd0440f2b978423440489beeba03fd4
```

**Point to these lines in this order:**

| What to point at | What it proves |
| --- | --- |
| `"new_values_match": true` | The new inputs produce output that matches PyTorch within tolerance. |
| `"observed_kernel_dispatches": 1` | The compatible call really dispatched the saved Metal kernel. |
| `"changed_shape_fallback": true`, `"fp32_input_fallback": true`, `"gradient_fallback": true` | The changed shape, changed dtype, and gradient call each used PyTorch. |
| `"model_requests": 0` | Reusing the saved kernel required no Muse call. |

**Say:** “I'm running fresh input values through the saved kernel. The result
matches PyTorch, and the audit saw exactly one Metal dispatch. The changed
shape, dtype, and gradient calls all went back to PyTorch. Reuse took zero
model requests.”

**If the command fails during rehearsal:** Show [GitHub → saved reuse audit,
lines 5–11](https://github.com/ramikamel/metalmind/blob/fff73af999b9c0591d58d46d45e5b112e98c6619/results/2026-11-04-muse-prompt-ab/focused-repair/reuse.json#L5-L11).
Point to the same lines and start with “In this saved audit on the M3 Pro…”
Use past tense throughout this shot.

## 6. Show where the speedup carries over · 1:53–2:17

**Show first:** [GitHub → Whole-workflow benchmarks](https://github.com/ramikamel/metalmind/blob/main/RESULTS.md#whole-workflow-benchmarks).
Point to the third table row: **FP16 attention block | Inductor split
attention | 1.058× / 1.067×**. The paragraph immediately below says this
used an earlier softmax artifact and includes both matrix multiplications.

**Show next:** [GitHub → Default checked mode](https://github.com/ramikamel/metalmind/blob/main/RESULTS.md#default-checked-mode).
Point to the opening sentence, **“All 108 checked-mode FP32 jobs kept
PyTorch.”**

**Say:** “An earlier verified softmax kernel helped a fixed attention block
run about 1.06 times as fast as Inductor, including both matrix
multiplications. With the default input scan included, the separate FP32
campaign kept PyTorch in all 108 jobs. The scan alone took longer than a
full PyTorch call in eight of nine measured FP32 signatures.”

## 7. Close on the headline results · 2:17–2:30

**Show:** [GitHub README → Benchmark results](https://github.com/ramikamel/metalmind#benchmark-results).
Keep the heading and table on screen. Point to **“9 of 9 signatures
promoted”** in the FP16 row and **“4 of 9 signatures promoted”** in the
FP32 row. Hold there through the last sentence.

**Say:** “With caller-guaranteed input bounds, MetalMind found verified
kernels for nine of nine FP16 signatures and four of nine FP32 signatures.
The source, failures, and timings are all in this repository.”

## After recording

Stop from the menu bar. Trim idle waits and play the export with sound to
check that the highlighted lines are readable. Use the video format or link
requested by the [submission portal](https://cerebralvalley.ai/e/meta-ai-developer-hackathon/hackathon/submit).

If you need a shorter cut, keep shots 1, 3, 4, and 5. Keep the shape, hardware,
and caller-guaranteed bounds in the narration.
