# MetalMind demo media

- [Video](video/metalmind-evidence-walkthrough.mp4) and [poster](video/poster.png)
- [Captions](video/captions.vtt) and [optional narration](video/voiceover.md)
- [Actual CLI recording](recording/demo.cast) and [plain transcript](recording/transcript.txt)
- [Structured replay](recording/demo.json), [capture metadata](recording/capture.json) and [render metadata](video/render.json)

The silent 1080p video runs for 179.96 seconds at 24 fps. Its ten paced chapters
walk through saved performance evidence from the 14-core Apple M3 Pro pilot
and the later whole-module comparison. The terminal capture also ran a fresh
correctness check on the promoted bias/GELU kernel. The output matched PyTorch
on new values with a maximum absolute error of 2.98e-08. Changed shapes and
gradient mode used the original function. The capture made zero model requests
and performed no fresh performance measurement.

The metadata binds the capture to source commit `209c82e` and the package
bytes actually executed. It records hashes for the terminal capture, replay
payload, transcript, video, captions and poster. The [demo guide](../docs/demo.md)
includes the reproduction commands.
