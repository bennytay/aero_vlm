"""Pinned public detector configuration."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from wam_drones.detection.vocabulary import DetectionLabel
from wam_drones.net import file_sha256 as file_sha256


class DetectorModelConfig(BaseModel):
    """Versioned identity and class projection for one public detector."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    package: str = Field(min_length=1)
    package_version: str = Field(min_length=1)
    license: str = Field(min_length=1)
    checkpoint: str = Field(min_length=1)
    checkpoint_url: str = Field(min_length=1)
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_size_px: int = Field(gt=0)
    precision: str = Field(min_length=1)
    source_taxonomy: str = Field(min_length=1)
    target_vocabulary: str = Field(min_length=1)
    class_mapping: dict[str, DetectionLabel]


def load_model_config(path: Path) -> DetectorModelConfig:
    """Load a checked-in model definition."""
    return DetectorModelConfig.model_validate(
        yaml.safe_load(path.read_text(encoding="utf-8"))
    )


def verify_checkpoint(path: Path, expected_sha256: str) -> None:
    """Refuse a checkpoint whose bytes do not match the pinned identity."""
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(
            f"checkpoint SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
