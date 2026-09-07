"""Pinned, reviewable definitions for image-only VLM experiments."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class VLMModelConfig(BaseModel):
    """One immutable public model identity and deterministic decode policy."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str = Field(min_length=1)
    repository: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    architecture: str = Field(min_length=1)
    license: str = Field(min_length=1)
    max_new_tokens: int = Field(gt=0)
    do_sample: bool
    temperature: float = Field(ge=0.0)
    image_max_pixels: int = Field(gt=0)
    supports_schema_constrained_decoding: bool


def load_model_config(path: Path) -> VLMModelConfig:
    """Load a checked-in VLM definition without initializing its runtime."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return VLMModelConfig.model_validate(payload)
