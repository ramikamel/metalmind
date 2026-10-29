"""Initial immutable runtime contracts. Optimization contracts follow separately."""
from dataclasses import dataclass


@dataclass(frozen=True)
class BackendCapabilities:
    mps_built: bool
    mps_available: bool
    compile_shader: bool
    timed_events: bool


@dataclass(frozen=True)
class EnvironmentInfo:
    schema_version: int
    python_version: str
    torch_version: str | None
    os_version: str
    architecture: str
    capabilities: BackendCapabilities
    unavailable_reason: str | None


from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

class Contract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    schema_version: Literal[1] = 1

class TensorSpec(Contract):
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: Literal["float32", "float16"] = "float32"
    device: Literal["mps", "cpu"] = "mps"
    storage_offset: int = Field(default=0, ge=0)

    @field_validator("shape", "strides", mode="before")
    @classmethod
    def integer_dimensions(cls, values):
        if not isinstance(values,(list,tuple)) or any(type(v) is not int or v<0 for v in values):
            raise ValueError("dimensions must be nonnegative integers")
        return values

    @model_validator(mode="after")
    def dimensions_match(self):
        if not self.shape or len(self.strides)!=len(self.shape):
            raise ValueError("shape and strides must have equal nonzero rank")
        return self

class WorkloadSpec(Contract):
    workload_id: str
    family: Literal["bias_gelu", "rmsnorm", "softmax"]
    semantics_version: int = 1
    inputs: tuple[TensorSpec, ...]
    output: TensorSpec
    epsilon: float = Field(default=0.0, ge=0)
    accumulation_dtype: Literal["float32", "float64"] = "float32"
    reference_id: str = ""
    reference_hash: str = ""
    policy_version: int = 1

    @model_validator(mode="after")
    def coherent_function(self):
        count=1 if self.family=="softmax" else 2
        if len(self.inputs)!=count or len(self.inputs[0].shape)!=2:
            raise ValueError("unsupported input count or rank")
        if any(d==0 or d>=2**32 for d in self.inputs[0].shape):
            raise ValueError("optimized shapes must be positive uint32 dimensions")
        if len(self.inputs)>1 and self.inputs[1].shape!=(self.inputs[0].shape[-1],):
            raise ValueError("parameter must match final dimension")
        if self.output.shape!=self.inputs[0].shape:
            raise ValueError("output shape must match input")
        if any(s.dtype!=self.output.dtype for s in self.inputs):
            raise ValueError("input and output dtypes must agree")
        return self

class CandidateProposal(Contract):
    entry_point: Literal["metalmind_kernel"] = "metalmind_kernel"
    source: str = Field(min_length=1, max_length=65536)
    threads: tuple[int,int,int]
    group_size: tuple[int,int,int]
    hypothesis: str = Field(min_length=1, max_length=2048)

    @field_validator("threads", "group_size", mode="before")
    @classmethod
    def launch_dimensions(cls,values):
        if not isinstance(values,(tuple,list)) or len(values)!=3 or any(type(v) is not int or v<=0 or v>=2**32 for v in values):
            raise ValueError("launch dimensions must be three positive uint32 integers")
        return values

class CandidateSpec(Contract):
    candidate_id: str
    workload_hash: str
    proposal: CandidateProposal
    parent_id: str | None = None
    model: str = "manual"

class VerificationSummary(Contract):
    passed: bool
    cases: int = Field(default=0, ge=0)
    max_absolute_error: float = Field(default=0, ge=0)
    max_relative_error: float = Field(default=0, ge=0)
    rmse: float = Field(default=0, ge=0)
    mismatches: int = Field(default=0, ge=0)
    reason: str | None = None

class EvaluationResult(Contract):
    status: Literal["valid", "compile_error", "runtime_error", "incorrect", "timeout", "cancelled"]
    verification: VerificationSummary | None = None
    elapsed_ms: tuple[float,...] = ()
    reason: str | None = None

    @model_validator(mode="after")
    def no_invalid_score(self):
        if self.elapsed_ms and (self.status!="valid" or self.verification is None or not self.verification.passed):
            raise ValueError("invalid candidates cannot receive timing scores")
        if any(t<=0 for t in self.elapsed_ms):
            raise ValueError("timings must be positive")
        return self
