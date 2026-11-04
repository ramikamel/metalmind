# Demonstrating MetalMind

Use the [timed recording guide](demo-recording-guide.md) for the current FP16
softmax demo. The [project walkthrough](project-flow.md) follows the full flow.
The material below documents the earlier recorded walkthrough.

[Watch the three-minute walkthrough](../demo/video/metalmind-evidence-walkthrough.mp4)
or read the [recorded terminal transcript](../demo/recording/transcript.txt).
The video is a silent, paced walkthrough with [captions](../demo/video/captions.vtt)
and an optional [voiceover script](../demo/video/voiceover.md). Its performance
numbers come from saved experiments. The capture also includes a fresh Apple
Silicon correctness check.

## Run the demonstration

From the installed checkout:

```sh
metalmind demo
metalmind demo --gpu-check
```

The first command validates the matching committed evidence on a CPU-only
machine. Pass `--evidence-root /path/to/metalmind/results` when running outside
the checkout. The optional GPU check exercises the promoted bias/GELU artifact
on new values and verifies original-function fallback for a changed shape and
gradient mode. Both commands use zero model requests.

1. Show the supported PyTorch bias/GELU function and explain the compile,
   correctness, timing and model-feedback loop.
2. Show the saved valid and rejected candidates in their recorded order. The
   rejected candidate produced non-finite output. Point out the generated Metal
   source and the private correctness checks that follow search.
3. Walk through all nine pilot signatures. Two won against verified function
   baselines and seven recorded no win. The medium FP32 bias/GELU signature
   measured 1.547× and 1.553× against Inductor fullgraph on a 14-core M3 Pro,
   under explicit assumed-domain bounds.
4. Show the fresh reuse check and the two fallback decisions.
5. Explain the module counterexample. The guarded regional rewrite beat eager
   but lost against the verified whole-module compiler path. The stricter
   default gate therefore kept the original module.

See the [function pilot](../results/2026-11-02-fp32-pilot/README.md) and
[whole-module comparison](../results/2026-11-02-module-baselines/README.md)
for raw pairs, scope and baseline identities. The recorded presentation uses
18 seconds per chapter for readability. See [event alignment](hackathon.md)
for the remaining authenticated submission steps.

## Reproduce the media

Use a clean, installed checkout with the matching committed evidence. Install
the optional rendering dependencies and choose two new output directories:

```sh
python -m pip install -r requirements-demo.lock
python -m scripts.record_demo --evidence-root results --output-dir runs/demo/recording --gpu-check
python -m scripts.render_demo --recording-dir runs/demo/recording --output-dir runs/demo/video --seconds-per-chapter 18
```

Omit `--gpu-check` for a CPU-only replay capture. The recorder saves actual CLI
output with timestamps and verifies the installed package bytes against the
committed source. The renderer uses the structured CLI output for its chapter
frames. See the [media index](../demo/README.md) for provenance and playback files.
