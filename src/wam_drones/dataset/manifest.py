"""Versioned provenance manifests for converted aerial datasets."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MANIFEST_VERSION: Literal["v1"] = "v1"

DatasetName = Literal["visdrone_det", "visdrone_mot", "visdrone_vid", "uavdt"]
DatasetSplit = Literal["train", "val", "test"]


class ImageRecord(BaseModel):
    """Provenance for one converted image; box geometry lives alongside it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: DatasetName
    split: DatasetSplit
    source_url: str = Field(min_length=1)
    original_id: str = Field(min_length=1)
    relative_image_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    sequence_id: str | None = Field(default=None, min_length=1)
    frame_number: int | None = Field(default=None, ge=0)
    annotation_conversion_version: str = Field(min_length=1)
    relative_label_path: str | None = Field(default=None, min_length=1)
    relative_annotation_path: str | None = Field(default=None, min_length=1)
    box_count: int = Field(ge=0)
    ignored_region_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_sequence_pairing(self) -> ImageRecord:
        """Require sequence_id and frame_number to be present together."""
        if (self.sequence_id is None) != (self.frame_number is None):
            raise ValueError(
                "sequence_id and frame_number must both be set or both be absent"
            )
        return self


class DatasetManifest(BaseModel):
    """Reproducible collection of provenance rows for one dataset split."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["v1"] = MANIFEST_VERSION
    dataset: DatasetName
    split: DatasetSplit
    created_utc: datetime
    samples: tuple[ImageRecord, ...]

    @model_validator(mode="after")
    def validate_samples(self) -> DatasetManifest:
        """Keep every row consistent with the manifest and with each other."""
        seen_ids: set[str] = set()
        seen_paths: set[str] = set()
        for sample in self.samples:
            if sample.dataset != self.dataset:
                raise ValueError("every sample must match the manifest dataset")
            if sample.split != self.split:
                raise ValueError("every sample must match the manifest split")
            if sample.original_id in seen_ids:
                raise ValueError(f"duplicate original_id: {sample.original_id}")
            if sample.relative_image_path in seen_paths:
                raise ValueError(
                    f"duplicate relative_image_path: {sample.relative_image_path}"
                )
            seen_ids.add(sample.original_id)
            seen_paths.add(sample.relative_image_path)
        return self

    def write(self, path: Path) -> None:
        """Write the manifest as indented, stably ordered JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            self.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def read(cls, path: Path) -> DatasetManifest:
        """Load and validate a previously written manifest."""
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
