# Verified development environment

Recorded October 29, 2026. These are observed capabilities on this machine,
not minimum requirements or a cross-device performance claim.

```json
{
  "schema_version": 1,
  "python_version": "3.14.6",
  "torch_version": "2.15.0",
  "os_version": "27.2",
  "architecture": "arm64",
  "capabilities": {
    "mps_built": true,
    "mps_available": true,
    "compile_shader": true,
    "timed_events": true
  },
  "unavailable_reason": null,
  "shader_health_passed": true,
  "probe_pipeline_max_threads": 1024
}
```

The GPU integration tests cover real compilation, mixed scalar arguments,
irregular element counts, repeated launches, typed invalid-source/entry errors,
CPU-buffer rejection, pipeline threadgroup bounds, health, and timed events.
Twelve tests passed, with no GPU skips, outside the restrictive execution sandbox.

Use `requirements-dev.lock` for the tested dependency snapshot. Python 3.14.6
was already installed and is compatible with the pinned PyTorch wheel here;
the design's suggested Python 3.12 is not required for this initial foundation.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.lock
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/metalmind doctor --json --smoke
.venv/bin/python -m pytest -q
```

`timed_events` in diagnostics describes API presence. The integration test
separately verifies positive elapsed time around completed GPU work.
Shader validation and general performance benchmarking are not implemented yet.
Meta access was subsequently verified with the key in the ignored local .env:
a bounded muse-wonder-1.0 request completed and returned the expected output.
The key was not displayed or committed. See progress.md for sanitized request
metadata. No optimized-kernel performance result is claimed.
