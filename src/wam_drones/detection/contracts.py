"""Versioned runtime contracts for aerial detection and tracking."""

from collections.abc import Sequence
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from wam_drones.detection.vocabulary import (
    DETECTION_LABEL_BY_ID,
    DetectionLabel,
)

CONTRACT_VERSION: Final = "v1"
TRACK_STALE_AFTER_FRAMES: Final = 3

NormalizedCoordinate = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
NormalizedBoundingBox = tuple[
    NormalizedCoordinate,
    NormalizedCoordinate,
    NormalizedCoordinate,
    NormalizedCoordinate,
]
NormalizedVelocity = tuple[
    Annotated[float, Field(allow_inf_nan=False)],
    Annotated[float, Field(allow_inf_nan=False)],
]


class FrozenContract(BaseModel):
    """Immutable contract base that rejects unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _validate_label_pair(label_id: int, label: DetectionLabel) -> None:
    expected = DETECTION_LABEL_BY_ID.get(label_id)
    if expected is None or label != expected:
        raise ValueError("label_id and label must match detection vocabulary v1")


def _validate_bbox(box: NormalizedBoundingBox) -> None:
    x_min, y_min, x_max, y_max = box
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("bounding box must have positive width and height")


class Detection(FrozenContract):
    """One fresh model detection in one captured frame."""

    contract_version: Literal["v1"] = CONTRACT_VERSION
    frame_id: int = Field(ge=0)
    captured_at_monotonic_ns: int = Field(ge=0)
    label_id: int = Field(ge=1, le=10)
    label: DetectionLabel
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    bbox_norm_xyxy: NormalizedBoundingBox
    model_name: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_detection(self) -> "Detection":
        """Keep labels and geometry internally consistent."""
        _validate_label_pair(self.label_id, self.label)
        _validate_bbox(self.bbox_norm_xyxy)
        return self


class FrameDetections(FrozenContract):
    """All fresh detections produced for one camera frame."""

    contract_version: Literal["v1"] = CONTRACT_VERSION
    frame_id: int = Field(ge=0)
    captured_at_monotonic_ns: int = Field(ge=0)
    inference_started_at_monotonic_ns: int = Field(ge=0)
    produced_at_monotonic_ns: int = Field(ge=0)
    image_width_px: int = Field(gt=0)
    image_height_px: int = Field(gt=0)
    detections: tuple[Detection, ...] = ()

    @model_validator(mode="after")
    def validate_frame(self) -> "FrameDetections":
        """Reject time reversal or detections from a different frame."""
        if not (
            self.captured_at_monotonic_ns
            <= self.inference_started_at_monotonic_ns
            <= self.produced_at_monotonic_ns
        ):
            raise ValueError("frame timestamps must be monotonic")
        for detection in self.detections:
            if detection.frame_id != self.frame_id:
                raise ValueError("every detection must match the frame_id")
            if detection.captured_at_monotonic_ns != self.captured_at_monotonic_ns:
                raise ValueError("every detection must match the frame capture time")
        return self


class TrackObservation(FrozenContract):
    """One tracked object, fresh or motion-predicted, in one frame."""

    contract_version: Literal["v1"] = CONTRACT_VERSION
    frame_id: int = Field(ge=0)
    captured_at_monotonic_ns: int = Field(ge=0)
    track_id: int = Field(ge=1)
    label_id: int = Field(ge=1, le=10)
    label: DetectionLabel
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    bbox_norm_xyxy: NormalizedBoundingBox
    velocity_norm_per_s: NormalizedVelocity
    age_frames: int = Field(ge=1)
    hits: int = Field(ge=1)
    time_since_update_frames: int = Field(ge=0)
    observed_this_frame: bool
    stale: bool

    @model_validator(mode="after")
    def validate_track(self) -> "TrackObservation":
        """Keep identity, geometry, lifecycle, and staleness consistent."""
        _validate_label_pair(self.label_id, self.label)
        _validate_bbox(self.bbox_norm_xyxy)
        if self.hits > self.age_frames:
            raise ValueError("track hits cannot exceed track age")
        if self.time_since_update_frames >= self.age_frames:
            raise ValueError("time since update must be less than track age")
        expected_observed = self.time_since_update_frames == 0
        if self.observed_this_frame != expected_observed:
            raise ValueError("observed_this_frame must match time since update")
        expected_stale = self.time_since_update_frames >= TRACK_STALE_AFTER_FRAMES
        if self.stale != expected_stale:
            raise ValueError("stale must match the configured frame threshold")
        return self


class FrameTracks(FrozenContract):
    """All active track observations for one camera frame."""

    contract_version: Literal["v1"] = CONTRACT_VERSION
    frame_id: int = Field(ge=0)
    captured_at_monotonic_ns: int = Field(ge=0)
    produced_at_monotonic_ns: int = Field(ge=0)
    tracks: tuple[TrackObservation, ...] = ()

    @model_validator(mode="after")
    def validate_frame(self) -> "FrameTracks":
        """Reject time reversal, mixed frames, or duplicate active IDs."""
        if self.produced_at_monotonic_ns < self.captured_at_monotonic_ns:
            raise ValueError("frame timestamps must be monotonic")
        track_ids: set[int] = set()
        for track in self.tracks:
            if track.frame_id != self.frame_id:
                raise ValueError("every track must match the frame_id")
            if track.captured_at_monotonic_ns != self.captured_at_monotonic_ns:
                raise ValueError("every track must match the frame capture time")
            if track.track_id in track_ids:
                raise ValueError("track IDs must be unique within a frame")
            track_ids.add(track.track_id)
        return self


DetectionFrame = FrameDetections | FrameTracks


def validate_frame_sequence(frames: Sequence[DetectionFrame]) -> None:
    """Require strictly increasing frame IDs and capture times."""
    previous_frame_id: int | None = None
    previous_timestamp: int | None = None
    for frame in frames:
        if previous_frame_id is not None and frame.frame_id <= previous_frame_id:
            raise ValueError("frame IDs must be strictly increasing")
        if (
            previous_timestamp is not None
            and frame.captured_at_monotonic_ns <= previous_timestamp
        ):
            raise ValueError("frame capture times must be strictly increasing")
        previous_frame_id = frame.frame_id
        previous_timestamp = frame.captured_at_monotonic_ns
