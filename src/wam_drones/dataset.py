"""Versioned manifest contracts for public-image evaluation subsets."""

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from wam_drones.contracts import NormalizedBoundingBox
from wam_drones.vocabulary import TARGET_IDS_BY_LABEL, VOCABULARY_VERSION


class DatasetSample(BaseModel):
    """One licensed image mapped to the frozen vocabulary or no-target flag."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    image_path: str = Field(min_length=1)
    image_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    split: Literal["train", "val", "test"]
    source: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    licence: str = Field(min_length=1)
    no_target: bool
    target_label: str | None = None
    target_id: int | None = Field(default=None, ge=1, le=255)
    bbox_xyxy_normalized: NormalizedBoundingBox | None = None
    colour_verified: bool = False

    @model_validator(mode="after")
    def validate_target(self) -> "DatasetSample":
        if self.no_target:
            if self.target_label is not None or self.target_id is not None:
                raise ValueError(
                    "no_target samples must not carry a target label or ID"
                )
            if self.bbox_xyxy_normalized is not None:
                raise ValueError("no_target samples must not carry a target box")
            return self
        if self.target_label is None or self.target_id is None:
            raise ValueError("target samples require target_label and target_id")
        expected_id = TARGET_IDS_BY_LABEL.get(self.target_label)
        if expected_id is None or expected_id != self.target_id:
            raise ValueError("target label and ID must match vocabulary_v0")
        return self


class DatasetManifest(BaseModel):
    """Reproducible collection of scene-separated public evaluation images."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["v0"] = VOCABULARY_VERSION
    created_utc: datetime
    vocabulary_path: str = "configs/vocabulary_v0.yaml"
    sources_path: str = "data/manifests/sources_v0.yaml"
    samples: tuple[DatasetSample, ...]

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> "DatasetManifest":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
