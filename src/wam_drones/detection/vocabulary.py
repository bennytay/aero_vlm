"""Frozen detection vocabulary and YAML loader."""

from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class DetectionLabel(StrEnum):
    """Version 1 aerial detection class names."""

    PEDESTRIAN = "pedestrian"
    PERSON = "person"
    BICYCLE = "bicycle"
    CAR = "car"
    VAN = "van"
    TRUCK = "truck"
    TRICYCLE = "tricycle"
    AWNING_TRICYCLE = "awning-tricycle"
    BUS = "bus"
    MOTOR = "motor"


DETECTION_LABEL_BY_ID: dict[int, DetectionLabel] = {
    index: label for index, label in enumerate(DetectionLabel, start=1)
}
DETECTION_ID_BY_LABEL: dict[DetectionLabel, int] = {
    label: index for index, label in DETECTION_LABEL_BY_ID.items()
}


class VocabularyLabel(BaseModel):
    """One stable numeric and textual class identifier."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: int = Field(ge=1, le=255)
    name: DetectionLabel


class DetectionVocabulary(BaseModel):
    """Complete version 1 aerial detection taxonomy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["v1"]
    source_taxonomy: Literal["VisDrone"]
    labels: tuple[VocabularyLabel, ...]

    @model_validator(mode="after")
    def validate_frozen_mapping(self) -> "DetectionVocabulary":
        """Reject missing, duplicate, reordered, or renamed classes."""
        actual = tuple((label.id, label.name) for label in self.labels)
        expected = tuple(DETECTION_LABEL_BY_ID.items())
        if actual != expected:
            raise ValueError("labels must exactly match detection vocabulary v1")
        return self


def load_detection_vocabulary(path: Path) -> DetectionVocabulary:
    """Load and validate a version 1 detection vocabulary."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return DetectionVocabulary.model_validate(payload)
