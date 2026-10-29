"""Validated nonsecret run configuration; credentials stay outside artifacts."""
import os
from pathlib import Path
from typing import Literal
import yaml
from dotenv import dotenv_values
from pydantic import Field, SecretStr, model_validator
from metalmind.contracts import Contract

class RunConfig(Contract):
    model: str = "muse-wonder-1.0"
    reasoning_effort: Literal["minimal","low","medium","high"] = "low"
    domain_mode: Literal["checked","assume"] = "checked"
    max_candidates: int = Field(default=30, gt=0)
    max_requests: int = Field(default=60, gt=0)
    wall_seconds: float = Field(default=1800, gt=0)
    finalization_reserve_seconds: float = Field(default=300, gt=0)
    max_output_tokens: int = Field(default=4096, ge=256, le=16384)
    total_output_tokens: int = Field(default=120000, gt=0)
    request_timeout_seconds: float = Field(default=120, gt=0)
    compile_timeout_seconds: float = Field(default=60, gt=0)
    execution_timeout_seconds: float = Field(default=30, gt=0)
    stagnation_limit: int = Field(default=10, gt=0)
    seed: int = Field(default=42, ge=0)
    runs_dir: Path = Path("runs")
    artifacts_dir: Path = Path("artifacts")
    warmups: int = Field(default=25, ge=25)
    search_pairs: int = Field(default=10, ge=2)
    confirmation_pairs: int = Field(default=30, ge=30)
    bootstrap_resamples: int = Field(default=10000, ge=1000)
    confirmation_cooldown_seconds: float = Field(default=60, ge=60)

    @model_validator(mode="after")
    def reserve_fits(self):
        if self.finalization_reserve_seconds>=self.wall_seconds:
            raise ValueError("finalization reserve must fit inside job wall limit")
        return self

def load_config(path: Path | None = None) -> RunConfig:
    values = yaml.safe_load(path.read_text()) if path else {}
    return RunConfig.model_validate(values or {})

def load_api_key(path: Path = Path(".env")) -> SecretStr:
    value=os.environ.get("MODEL_API_KEY") or dotenv_values(path).get("MODEL_API_KEY")
    if not value:
        raise ValueError("Set MODEL_API_KEY in the environment or local .env")
    return SecretStr(value)
