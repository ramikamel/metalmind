import pytest
import torch
from examples import fp16_workloads
from metalmind.workloads.registry import get_workload,resolve_reference,resolve_oracle
from metalmind.workloads.fixtures import make_public_fixtures

@pytest.mark.parametrize("family",["bias_gelu","rmsnorm","softmax"])
def test_fp16_uses_float_intermediates_and_final_cast(family):
    spec=get_workload(family,"small","float16")
    fn=resolve_reference(spec.reference_id)
    for fixture in make_public_fixtures(spec):
        output=fn(*fixture.inputs)
        assert output.dtype==torch.float16
        torch.testing.assert_close(output,resolve_oracle(family)(*fixture.inputs).half(),atol=.002,rtol=.02)


def rounded_bias_add(x, bias):
    return torch.nn.functional.gelu((x + bias).float(), approximate="tanh").half()


def wide_softmax_output(x):
    return torch.softmax(x.float(), dim=-1)


@pytest.mark.parametrize("family,fn", [
    ("bias_gelu", fp16_workloads.bias_gelu),
    ("rmsnorm", fp16_workloads.rmsnorm),
    ("softmax", fp16_workloads.row_softmax),
])
def test_capture_admits_only_the_explicit_fp16_reference(family, fn):
    from metalmind.workloads.capture import UnsupportedReason, capture_function

    registered = get_workload(family, "small", "float16")
    inputs = make_public_fixtures(registered)[0].inputs
    result = capture_function(fn, inputs)
    assert not isinstance(result, UnsupportedReason), getattr(result, "reason", None)
    assert result.family == family
    assert result.inputs == registered.inputs
    assert result.output.dtype == "float16"
    assert result.accumulation_dtype == "float32"
    torch.testing.assert_close(fn(*inputs), resolve_reference(family)(*inputs), rtol=0, atol=0)


@pytest.mark.parametrize("family,fn", [
    ("bias_gelu", rounded_bias_add),
    ("softmax", wide_softmax_output),
])
def test_capture_rejects_different_fp16_rounding_or_output(family, fn):
    from metalmind.workloads.capture import UnsupportedReason, capture_function

    spec = get_workload(family, "small", "float16")
    assert isinstance(capture_function(fn, make_public_fixtures(spec)[0].inputs), UnsupportedReason)


def test_capture_rejects_mixed_input_dtypes():
    from metalmind.workloads.capture import UnsupportedReason, capture_function

    x = torch.zeros((2, 3), dtype=torch.float16)
    bias = torch.zeros(3, dtype=torch.float32)
    assert isinstance(capture_function(fp16_workloads.bias_gelu, (x, bias)), UnsupportedReason)


def test_explicit_widening_avoids_rounding_bias_sum_in_half():
    x = torch.tensor([[0.5]], dtype=torch.float16)
    bias = torch.tensor([0.50048828125], dtype=torch.float16)
    assert fp16_workloads.bias_gelu(x, bias).item() == 0.841796875
    assert rounded_bias_add(x, bias).item() == 0.84130859375


@pytest.mark.parametrize("reference,reason", [
    ("python:examples.fp16_workloads:bias_gelu", "backend reached"),
    ("python:test_fp16_semantics:rounded_bias_add", "no longer matches supported semantics"),
])
def test_worker_rechecks_fp16_cast_order_before_loading_backend(reference, reason, monkeypatch):
    from metalmind.agent.candidates import make_candidate
    from metalmind.contracts import EvaluationRequest
    from metalmind.evaluation import template_proposals
    from metalmind.runtime.worker import _child
    from metalmind.workloads.registry import fingerprint_reference

    def backend():
        raise ValueError("backend reached")

    class Connection:
        messages = []

        def send(self, message):
            self.messages.append(message)

        def close(self):
            pass

    monkeypatch.setattr("metalmind.runtime.metal.MetalBackend", backend)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    workload = get_workload("bias_gelu", "small", "float16").model_copy(update={
        "reference_id": reference, "reference_hash": fingerprint_reference(reference),
    })
    candidate = make_candidate(template_proposals(workload)[0].proposal, workload)
    connection = Connection()
    _child(EvaluationRequest(candidate=candidate, workload=workload).model_dump(), connection)
    assert reason in connection.messages[0]["result"]["reason"]


def string_narrowing(x):
    return torch.softmax(x.float(), dim=-1).to("torch.float16")


def test_cast_device_string_cannot_impersonate_a_dtype():
    from metalmind.workloads.capture import UnsupportedReason, capture_function

    spec = get_workload("softmax", "small", "float16")
    result = capture_function(string_narrowing, make_public_fixtures(spec)[0].inputs)
    assert isinstance(result, UnsupportedReason)


def keyword_only_softmax(*, x):
    return torch.softmax(x.float(), dim=-1).half()


def wrong_keepdim_type(x, weight):
    z = x.float()
    return (z * torch.rsqrt(z.square().mean(dim=-1, keepdim=1) + 1e-5) * weight.float()).half()


def invalid_unused_reduction(x):
    z = x.float()
    unused = z.mean(dim=9)
    return torch.softmax(z, dim=-1).half()


@pytest.mark.parametrize("family,fn", [
    ("softmax", keyword_only_softmax),
    ("rmsnorm", wrong_keepdim_type),
    ("softmax", invalid_unused_reduction),
])
def test_fp16_capture_rejects_references_that_fail_the_call_contract(family, fn):
    from metalmind.workloads.capture import UnsupportedReason, capture_function

    spec = get_workload(family, "small", "float16")
    assert isinstance(capture_function(fn, make_public_fixtures(spec)[0].inputs), UnsupportedReason)


def rounded_rmsnorm(x, weight):
    scale = torch.rsqrt(x.square().float().mean(dim=-1, keepdim=True) + 1e-5)
    return (x.float() * scale * weight.float()).half()


def test_fp16_rmsnorm_rejects_squaring_before_widening():
    from metalmind.workloads.capture import UnsupportedReason, capture_function

    x = torch.tensor([[45.25, 0.3, -3.4]], dtype=torch.float16)
    weight = torch.ones(3, dtype=x.dtype)
    assert not torch.equal(fp16_workloads.rmsnorm(x, weight), rounded_rmsnorm(x, weight))
    assert isinstance(capture_function(rounded_rmsnorm, (x, weight)), UnsupportedReason)
