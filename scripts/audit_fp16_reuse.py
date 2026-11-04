"""Audit a registered FP16 artifact offline with observed Metal dispatch counts."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import torch

from metalmind.artifacts.export import verify_artifact_evidence
from metalmind.artifacts.loader import load_artifact
from metalmind.runtime.lock import gpu_lock
from metalmind.runtime.metal import MetalBackend
from metalmind.runtime.worker import gpu_faulted
from metalmind.verification.numerics import inputs_in_domain
from metalmind.verification.policy import policy_for
from metalmind.workloads.registry import resolve_reference


def audit(path):
    manifest = verify_artifact_evidence(path)
    workload = manifest.workload
    if workload.reference_id != workload.family or workload.inputs[0].dtype != "float16":
        raise ValueError("This audit requires a registered FP16 artifact")
    if gpu_faulted():
        raise RuntimeError("Run the Metal health check before auditing")
    reference = resolve_reference(workload.reference_id)
    policy = policy_for(workload)
    dispatches = 0
    compile_kernel = MetalBackend.compile

    def observe_compile(backend, *args, **kwargs):
        kernel = compile_kernel(backend, *args, **kwargs)

        def observed_dispatch(*inputs, **options):
            nonlocal dispatches
            dispatches += 1
            return kernel.function(*inputs, **options)

        return replace(kernel, function=observed_dispatch)

    with gpu_lock(), patch.object(MetalBackend, "compile", observe_compile):
        optimized = load_artifact(path, fallback=reference)
        if optimized.optimized is None:
            raise ValueError(optimized.reason)
        generator = torch.Generator().manual_seed(2718)
        inputs = tuple(
            (torch.randint(-32, 33, spec.shape, generator=generator).half() / 16).to("mps")
            for spec in workload.inputs
        )
        if not inputs_in_domain(workload, inputs):
            raise ValueError("Audit inputs are outside the artifact domain")
        with torch.no_grad():
            actual, expected = optimized(*inputs), reference(*inputs)
            torch.testing.assert_close(actual, expected, atol=policy.atol, rtol=policy.rtol)
            error = (actual.float() - expected.float()).abs().max().item()
            if dispatches != 1:
                raise AssertionError("Exact FP16 inputs did not dispatch the promoted kernel")
            x, *parameters = inputs
            mismatches = (
                (torch.cat((x, x[:1])), *parameters),
                tuple(t.float() for t in inputs),
            )
            for args in mismatches:
                torch.testing.assert_close(optimized(*args), reference(*args), rtol=0, atol=0)
        grad_input = x.detach().requires_grad_(True)
        optimized(grad_input, *parameters).sum().backward()
        if grad_input.grad is None or dispatches != 1:
            raise AssertionError("Shape, dtype or gradient fallback dispatched the Metal artifact")
    return {
        "artifact_id": manifest.artifact_id,
        "dtype": workload.inputs[0].dtype,
        "domain_mode": manifest.domain_mode,
        "new_values_match": True,
        "changed_shape_fallback": True,
        "fp32_input_fallback": True,
        "gradient_fallback": True,
        "observed_kernel_dispatches": dispatches,
        "max_absolute_error": error,
        "model_requests": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.artifact), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
