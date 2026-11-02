import pytest
import torch
from examples.integrated_block import build_workflow
from metalmind.evaluation import template_proposals
from metalmind.runtime.execution import build_execution_callable
from metalmind.runtime.metal import MetalBackend
from metalmind.workloads.registry import workload_for_shape, resolve_reference

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")]


def test_generated_activation_preserves_whole_matmul_workflow():
    workload = workload_for_shape("bias_gelu", (8, 257))
    reference = resolve_reference("bias_gelu")
    proposal = template_proposals(workload)[0].proposal
    kernel = MetalBackend().compile(proposal.source, entry_point="metalmind_kernel")
    activation = build_execution_callable(workload, kernel, fallback=reference, domain_mode="assume")
    activation.proposal = proposal
    for _ in range(2):
        original, optimized = build_workflow(workload, reference, activation)
        with torch.inference_mode():
            torch.testing.assert_close(optimized(), original(), atol=1e-6, rtol=1e-5)
