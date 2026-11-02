import json
import pytest
from pydantic import ValidationError
from metalmind.contracts import WorkloadSpec, CandidateProposal, EvaluationResult
from metalmind.config import RunConfig, load_config


def test_workload_roundtrip_and_frozen(sample_workload):
    assert WorkloadSpec.model_validate_json(sample_workload.model_dump_json()) == sample_workload
    with pytest.raises(ValidationError):
        sample_workload.family = "softmax"


@pytest.mark.parametrize("patch", [{"schema_version": 2}, {"output": None},
    {"inputs": [{"shape": [-1, 3], "strides": [3, 1]}]}, {"family": "unknown"}])
def test_invalid_workload_rejected(sample_workload, patch):
    with pytest.raises(ValidationError):
        WorkloadSpec.model_validate(sample_workload.model_dump() | patch)


def test_candidate_forbids_harness_fields():
    valid = dict(source="kernel void metalmind_kernel() {}", threads=(6,1,1), group_size=(2,1,1), hypothesis="fuse")
    CandidateProposal(**valid)
    for patch in ({"threads": (0,1,1)}, {"tolerance": 1.0}, {"group_size": (True,1,1)}):
        with pytest.raises(ValidationError):
            CandidateProposal(**(valid | patch))


def test_invalid_result_cannot_have_timings():
    with pytest.raises(ValidationError):
        EvaluationResult(status="incorrect", elapsed_ms=(1.0,))


def test_config_redaction_and_budget_validation(tmp_path):
    p=tmp_path / "config.yaml"; p.write_text("max_candidates: 3\ndomain_mode: assume\n")
    assert load_config(p).max_candidates == 3
    assert "api_key" not in RunConfig().model_dump_json()
    for patch in ({"wall_seconds": 20}, {"epsilon": float('nan')}, {"max_requests": 0}):
        with pytest.raises(ValidationError): RunConfig(**patch)


def test_uint32_element_count_cannot_overflow():
    from metalmind.contracts import TensorSpec
    tensor=TensorSpec(shape=(65536,65536),strides=(65536,1))
    parameter=TensorSpec(shape=(65536,),strides=(1,))
    with pytest.raises(ValidationError):
        WorkloadSpec(workload_id='oversized',family='bias_gelu',inputs=(tensor,parameter),output=tensor)
