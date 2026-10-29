# MetalMind

Autonomous, numerically verified Metal kernel optimization for PyTorch on Apple Silicon.

## Current status

Initial runtime foundation: environment diagnostics, direct shader compilation and dispatch,
known-good GPU health check, and real MPS integration tests. The optimization agent is planned;
no generated-kernel speedup is claimed yet.

## Development

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
metalmind doctor --json --smoke
python -m pytest -q
```

The pinned runtime is PyTorch 2.15.0. GPU tests skip explicitly without usable MPS.
On supported Apple Silicon, the smoke command compiles and executes a real Metal kernel.
GPU access can be unavailable inside restrictive sandboxes even on supported hardware.

## Design and roadmap

- [System design](docs/superpowers/specs/2026-10-29-metalmind-design.md)
- [Implementation plan](docs/superpowers/plans/2026-10-29-metalmind.md)
- [Implementation progress](docs/progress.md)

Start with explicitly supported functions, then add guarded graph-region replacement.
Correctness and synchronized measurements precede performance claims.
