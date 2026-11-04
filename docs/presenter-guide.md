# Present MetalMind

## Start here

The hackathon release is ready to present. The remaining work is to rehearse,
sign in to the event portal, check its current fields and media requirements,
and submit. You can use the recorded walkthrough for the demo. A live terminal
run is optional.

The open research targets are faster default checked-mode execution, a win
against compiled PyTorch for a whole module, and support for more models and
hardware. You can explain those as future work. The released project already
has a working end-to-end optimization loop and verified function-level wins.

1. Read the explanation below.
2. Play the [three-minute walkthrough](../demo/video/metalmind-evidence-walkthrough.mp4)
   and follow the narration by timestamp.
3. Open the [FP16 result table](../results/2026-11-03-assumed-fp16-pilot/README.md)
   in a second tab. It contains the strongest result from the later pilot.
4. Use the [submission draft](submission-draft.md) when the portal is open.

## What you built

> MetalMind uses a Meta model to propose Apple GPU code for a supported PyTorch
> function. It tests the code on a real GPU and keeps it only if the outputs
> match and the speedup holds up against PyTorch.

Imagine your PyTorch code adds a bias to a matrix and then applies GELU, an
activation used in neural networks. PyTorch already runs this on the Apple GPU.
MetalMind explores whether a custom implementation can do the same work
faster. For example, one GPU program can combine the addition and activation
and avoid writing an intermediate tensor.

**PyTorch** is the library the developer uses. **Metal** is Apple's GPU
programming layer. A **kernel** is one program that runs on the GPU.
**Muse Wonder** is the Meta model that writes candidate kernels. **Inductor**
is PyTorch's compiler and an important baseline because it can also fuse
operations.

### Follow one function through the system

1. **Describe the workload.** MetalMind starts with a supported function such
   as bias/GELU, RMSNorm, or row softmax. It records the operation, tensor
   shapes, dtype, layout, and input assumptions. For the medium FP32 bias/GELU
   example, the main input is a matrix with 256 rows and 1,024 columns. The
   kernel adds the same 1,024-element bias to each row, then applies GELU.
2. **Ask Meta for a candidate.** MetalMind sends Muse Wonder the workload,
   expected inputs and outputs, and constraints on how the GPU code can run.
   The model returns Metal source, launch dimensions, and an explanation of
   its approach. For these published experiments, the workload inputs are
   synthetic and the model gets the specification and evaluation feedback.
3. **Compile and verify locally.** MetalMind's Python code checks the proposal,
   compiles it, and runs it on the Mac's GPU. It compares every output element
   with the original PyTorch function within a dtype-specific tolerance.
   It also checks for problems such as unwritten output, modified inputs,
   and results that change across repeated calls.
4. **Measure and feed back.** Correct candidates are timed against verified
   PyTorch baselines. These can include ordinary PyTorch, a comparable built-in
   operation, and compiled PyTorch. MetalMind sends public errors and timings
   back to the model so it can use its remaining request budget to try a new
   candidate. The model supplies code. The Mac supplies the measurements.
5. **Confirm before keeping a winner.** MetalMind freezes the best eligible
   candidate, checks it on 22 private cases withheld from search, then runs
   two separate timing sessions. A winner must reach at least 1.05× and have
   a lower 95% interval bound above 1.00× against every required verified
   baseline in both sessions. If any gate fails, the original PyTorch
   function stays in use.
6. **Save and reuse.** A promoted artifact bundles the Metal source, expected
   input signature, launch settings, and verification evidence. MetalMind can
   load it offline and use it on new matching values. A mismatched shape or
   dtype, a gradient-enabled call with a tensor that requires gradients, or an
   incompatible environment takes the original PyTorch path. The offline reuse
   step makes zero model requests.

The system can also try replacing a supported region inside a small flat FP32
module. It measures the complete module separately before enabling that rewrite.
In the recorded example the region helped against ordinary execution, and the
compiled whole module was faster, so MetalMind kept the original module.
[Architecture](architecture.md) · [Module comparison](../results/2026-11-02-module-baselines/README.md)

## What it does with real PyTorch code

### One function

Here is a supported function. It converts the input to FP32 before doing
softmax and converts the finished result back to FP16.

```python
def row_softmax(x):
    return torch.softmax(x.float(), dim=-1).half()
```

Given an FP16 example with shape `(256, 1024)`, MetalMind recognizes the
calculation. It asks Muse Wonder to write a Metal implementation of row
softmax for that signature. It compares the proposed implementation with
PyTorch and keeps the original callable available for fallback. A winning
artifact can accept new values of the same shape and dtype. A new shape or
dtype needs its own optimization to use a custom kernel.

You can see the recognition step without making a model request:

```sh
.venv/bin/metalmind analyze --function examples.fp16_workloads:row_softmax \
  --workload softmax --case medium --dtype float16
```

### One region inside a larger module

The repository's module example has a forward pass like this:

```python
def forward(self, x):
    activation = F.gelu(x + self.bias, approximate="tanh")
    return activation + x * 0.25
```

MetalMind can identify the bias/GELU region in this supported flat FP32
module. If you supply a matching, previously promoted artifact, it can
try replacing that region with one Metal call. PyTorch still runs the
multiply and residual addition. The module path checks the complete
output and times the entire forward pass, including the unchanged
operations. In this actual example, the rewrite lost to compiled PyTorch,
so MetalMind returned the original module.

```sh
.venv/bin/metalmind analyze-module --factory examples.module_workloads:make_cpu_demo
```

A larger script can call an optimized function at a chosen point and keep
running all its other code normally. Automatic module analysis has a
narrower boundary. It handles supported pure regions in simple flat FP32
inference modules. Arbitrary programs with nested layers, Python loops,
training, or dynamic control flow fall outside that automatic workflow.
[Function examples](../examples/fp16_workloads.py) · [Module example](../examples/module_workloads.py)

## The numbers you can say out loud

| Experiment | Safe wording |
| --- | --- |
| Original FP32 pilot shown in the video | “Medium bias/GELU ran about **1.55× as fast as compiled PyTorch** in two confirmation sessions on our M3 Pro, when the caller guaranteed the input bounds.” |
| Later FP16 pilot | “**Six of nine** signatures promoted across all three workload families. Medium softmax reached **2.64× and 2.54× against the fastest verified baseline** in two sessions. These were FP16 calls with guaranteed input bounds.” |
| Default checked-mode FP32 comparison | “We ran **108 jobs** with the input scan included. Zero kernels promoted, so those jobs kept PyTorch. The results didn’t establish a promotion advantage for model feedback.” |
| Whole-module example | “The regional rewrite beat eager PyTorch, then lost to the compiled whole module. MetalMind kept the original.” |

**Checked mode** scans the contents of the inputs on every call to verify that
they meet the numerical bounds. **Assumed-domain mode** skips that contents
scan after the caller explicitly guarantees the bounds. Shape, dtype, device,
and other signature guards still apply. This distinction explains why the
conditional speedups and the default-mode results are both true.

“2.54× as fast” means the baseline took 2.54 times as long as the optimized call.
It refers to one exact function signature on the tested Apple M3 Pro. The
optimizer's search and first-load costs are separate from steady-state latency.
[Original pilot](../results/2026-11-02-fp32-pilot/README.md) · [FP16 pilot](../results/2026-11-03-assumed-fp16-pilot/README.md) · [Checked-mode comparison](../results/2026-11-03-checked-fp32-full/README.md)

## Read-aloud demo script

The [video](../demo/video/metalmind-evidence-walkthrough.mp4) is silent.
Read one row as each chapter comes up. The numbers on screen are from the
original FP32 pilot. The FP16 result can be your closing sentence or a
separate screenshot of the later results.

| Time | What to say |
| --- | --- |
| **0:00** | “MetalMind is an AI agent for optimizing PyTorch functions on Apple Silicon. Muse Wonder writes candidate GPU code. My system compiles it, checks the results, and measures whether it is faster.” |
| **0:18** | “We started with nine exact workloads on an Apple M3 Pro. Two passed every gate. The other seven kept the original PyTorch function. This first pilot used caller-guaranteed input bounds.” |
| **0:36** | “Here are two actual proposals from one search. One passed the public tests. The next produced non-finite output and was rejected. MetalMind records those outcomes and feeds public measurements back into the search.” |
| **0:54** | “A candidate that looks promising during search has to pass hidden test inputs and two independent timing sessions. This larger bias/GELU case passed, with a modest gain over compiled PyTorch.” |
| **1:12** | “The medium bias/GELU case ran about one and a half times as fast as compiled PyTorch. The benchmark includes the work a real call pays for, including allocation and GPU dispatch.” |
| **1:30** | “This is an excerpt of the Metal source Muse Wonder generated. The full kernel is saved with its launch settings, verification results, and raw timings, so the experiment can be inspected and replayed.” |
| **1:48** | “A fast function can still slow down a larger program. We ran a separate test on the entire module to see whether replacing just one region helped.” |
| **2:06** | “The rewrite beat ordinary PyTorch, but the compiled whole module was faster. MetalMind kept the original module. The acceptance rule is based on the best verified baseline we can run.” |
| **2:24** | “Here the saved kernel runs on new values without a model request. It matches PyTorch, and changing the shape or using gradients switches the call back to PyTorch.” |
| **2:42** | “A later FP16 pilot produced six verified wins, including medium softmax at about two and a half times the fastest verified baseline with guaranteed input bounds. Our separate checked-mode FP32 run kept PyTorch on all 108 jobs.” |

The [FP16 confirmation follow-up](../results/2026-11-03-fp16-confirmation-recheck/README.md)
later promoted small bias/GELU and small RMSNorm from the original saved
kernels. Large RMSNorm remained inconclusive after repeated baseline drift.
Use “six of nine” when describing the original pilot and “eight across the
pilot and follow-up” when describing the combined evidence.

## If you want a live terminal moment

On the same Mac, run this before recording to rehearse. The demo can play on
its own if the live correctness check is unavailable.

```sh
cd ~/github/metalmind
.venv/bin/metalmind demo
.venv/bin/metalmind demo --gpu-check
```

The first command replays saved results. The second additionally runs the
saved FP32 artifact on fresh inputs on the matching Mac. Neither command
calls Meta or measures new speed. If you use the second command on camera,
point at **Offline reuse check** and explain why the changed input shape
takes the fallback path. The paid `optimize` command is unnecessary for
this walkthrough.

## Likely judge questions

**What is autonomous?** After you select a supported workload, the controller
automatically requests code, compiles it, evaluates it, and uses the feedback
until the budget or stopping rule is reached. It decides whether to promote
or fall back. A developer still chooses the workload and deployment conditions.

**How do you know the speedup is real?** The GPU completes the work before
timing ends. We use paired measurements against PyTorch, private numerical
tests, and two separate confirmation sessions. The repository includes
the raw timings and the cases that failed.

**Why use an AI model if PyTorch already compiles?** Muse Wonder can propose
hardware-specific implementations and use compiler and benchmark feedback.
PyTorch's compiler is a competitor in the test and sometimes wins. The
benefit is an automated way to discover and verify where a custom kernel helps.

**Why was there no win in default mode?** The input contents scan is part of
that measured call. The checked experiment found zero promotions. The
separate assumed-domain pilots established gains for callers who already
guarantee the supported input bounds.

**What can you submit today?** The public repository, the silent walkthrough
or your narrated recording, and the [submission draft](submission-draft.md).
The authenticated portal supplies the exact fields and media limits. Sign in
through the [official submission portal](https://cerebralvalley.ai/e/meta-ai-developer-hackathon/hackathon/submit).
