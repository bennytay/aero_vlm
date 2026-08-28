"""Versioned desktop contracts shared by Phase 0 producers and consumers."""

from enum import StrEnum
from typing import Annotated, Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

TARGET_OBSERVATION_STALE_AFTER_MS: Final = 200

NormalizedCoordinate = Annotated[float, Field(ge=0.0, le=1.0)]
NormalizedBoundingBox = tuple[
    NormalizedCoordinate,
    NormalizedCoordinate,
    NormalizedCoordinate,
    NormalizedCoordinate,
]


class Behaviour(StrEnum):
    """Behaviours supported by the frozen L1 grammar."""

    FIND = "FIND"
    APPROACH = "APPROACH"
    HOLD = "HOLD"


class CommandState(BaseModel):
    """Structured result of a successfully parsed L1 command."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    command_id: str = Field(min_length=1)
    behaviour: Behaviour
    target_label: str = Field(min_length=1)
    target_id: int = Field(ge=1, le=255)
    desired_distance_m: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    command_timeout_ms: int | None = Field(default=None, gt=0)


class TargetObservation(BaseModel):
    """Version 0 target localisation result in normalised image coordinates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    timestamp_us: int = Field(ge=0)
    visible: bool
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    center_x_normalized: NormalizedCoordinate | None = None
    center_y_normalized: NormalizedCoordinate | None = None
    bbox_xyxy_normalized: NormalizedBoundingBox | None = None
    target_label: str | None = None
    target_id: int | None = Field(default=None, ge=1, le=255)
    stale: bool

    @model_validator(mode="after")
    def validate_box_order(self) -> "TargetObservation":
        """Reject inverted boxes while retaining nullable no-target observations."""
        if self.bbox_xyxy_normalized is not None:
            x_min, y_min, x_max, y_max = self.bbox_xyxy_normalized
            if x_min > x_max or y_min > y_max:
                raise ValueError("bounding box minimums must not exceed maximums")
        return self
