"""Phase 5 scheduling and tile-coordinate utilities.

These utilities deliberately sit below any model backend: the experiment can
exercise FP32, FP16, and target-native INT8 engines without changing cadence,
tiling, or the truthfulness of the tracking contract.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import yaml

from wam_drones.tracking.offline import Box, TrackingDetection, _iou

TilingMode = Literal["off", "scheduled_2x2", "uncertainty_2x2"]


@dataclass(frozen=True)
class EfficiencyProtocol:
    """The pre-registered Phase 5 matrix and its sealed sequence split."""

    checkpoint_path: str
    source_fps: float
    development_sequences: tuple[str, ...]
    held_out_sequences: tuple[str, ...]
    input_sizes_px: tuple[int, ...]
    precisions: tuple[str, ...]
    detector_cadences_frames: tuple[int, ...]
    tiling_policies: tuple[TilingPolicy, ...]
    trackers: tuple[str, ...]
    preview: tuple[bool, ...]

    def __post_init__(self) -> None:
        if self.source_fps <= 0:
            raise ValueError("source_fps must be positive")
        if not self.development_sequences or not self.held_out_sequences:
            raise ValueError(
                "Phase 5 requires non-empty development and held-out splits"
            )
        if set(self.development_sequences) & set(self.held_out_sequences):
            raise ValueError(
                "Phase 5 development and held-out sequences must be disjoint"
            )
        if not all(value > 0 for value in self.input_sizes_px):
            raise ValueError("input sizes must be positive")
        if not all(value > 0 for value in self.detector_cadences_frames):
            raise ValueError("detector cadences must be positive")


def load_efficiency_protocol(path: Path) -> EfficiencyProtocol:
    """Load the strict, versioned Phase 5 protocol from YAML."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Phase 5 protocol must be a YAML mapping")
    checkpoint = payload.get("checkpoint")
    matrix = payload.get("matrix")
    if not isinstance(checkpoint, Mapping) or not isinstance(matrix, Mapping):
        raise ValueError("Phase 5 protocol requires checkpoint and matrix mappings")
    tiling_payload = matrix.get("tiling")
    if not isinstance(tiling_payload, list):
        raise ValueError("Phase 5 matrix tiling must be a list")
    policies = []
    for item in tiling_payload:
        if not isinstance(item, Mapping):
            raise ValueError("each Phase 5 tiling policy requires a mode")
        raw_mode = item.get("mode")
        # YAML 1.1 treats the unquoted protocol value `off` as false.
        mode = "off" if raw_mode is False else raw_mode
        if not isinstance(mode, str):
            raise ValueError("each Phase 5 tiling policy requires a mode")
        if mode not in {"off", "scheduled_2x2", "uncertainty_2x2"}:
            raise ValueError(f"unknown Phase 5 tiling mode: {mode}")
        policies.append(
            TilingPolicy(
                mode=cast(TilingMode, mode),
                overlap_fraction=float(item.get("overlap_fraction", 0.20)),
                interval_frames=item.get("interval_frames"),
                no_confident_small_object_frames=item.get(
                    "no_confident_small_object_frames"
                ),
                small_box_max_area_norm=float(
                    item.get("small_box_max_area_norm", 0.0025)
                ),
                confidence_threshold=float(item.get("confidence_threshold", 0.50)),
            )
        )
    try:
        return EfficiencyProtocol(
            checkpoint_path=str(checkpoint["path"]),
            source_fps=float(payload["source_fps"]),
            development_sequences=tuple(
                str(item) for item in payload["development_sequences"]
            ),
            held_out_sequences=tuple(
                str(item) for item in payload["held_out_sequences"]
            ),
            input_sizes_px=tuple(int(item) for item in matrix["input_sizes_px"]),
            precisions=tuple(str(item) for item in matrix["precisions"]),
            detector_cadences_frames=tuple(
                int(item) for item in matrix["detector_cadences_frames"]
            ),
            tiling_policies=tuple(policies),
            trackers=tuple(str(item) for item in matrix["trackers"]),
            preview=tuple(bool(item) for item in matrix["preview"]),
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"invalid Phase 5 protocol: {path}") from error


def validate_efficiency_profile(
    protocol: EfficiencyProtocol,
    *,
    input_size_px: int,
    precision: str,
    detector_cadence_frames: int,
    tiling_policy: TilingPolicy,
    tracker_config: str,
    preview: bool,
) -> None:
    """Reject profiles absent from the pre-registered Phase 5 matrix."""
    if input_size_px not in protocol.input_sizes_px:
        raise ValueError(f"input size is not in the Phase 5 matrix: {input_size_px}")
    if precision not in protocol.precisions:
        raise ValueError(f"precision is not in the Phase 5 matrix: {precision}")
    if detector_cadence_frames not in protocol.detector_cadences_frames:
        raise ValueError("detector cadence is not in the Phase 5 matrix")
    if tiling_policy not in protocol.tiling_policies:
        raise ValueError("tiling policy is not in the Phase 5 matrix")
    if tracker_config not in protocol.trackers:
        raise ValueError("tracker configuration is not in the Phase 5 matrix")
    if preview not in protocol.preview:
        raise ValueError("preview selection is not in the Phase 5 matrix")


@dataclass(frozen=True)
class TilingPolicy:
    """A reproducible 2x2 tiling trigger for a full-frame detector stream."""

    mode: TilingMode
    overlap_fraction: float = 0.20
    interval_frames: int | None = None
    no_confident_small_object_frames: int | None = None
    small_box_max_area_norm: float = 0.0025
    confidence_threshold: float = 0.50

    def __post_init__(self) -> None:
        if self.mode not in {"off", "scheduled_2x2", "uncertainty_2x2"}:
            raise ValueError(f"unknown tiling mode: {self.mode}")
        if not 0.0 <= self.overlap_fraction < 0.5:
            raise ValueError("tile overlap_fraction must be in [0, 0.5)")
        if self.small_box_max_area_norm <= 0 or self.small_box_max_area_norm > 1:
            raise ValueError("small_box_max_area_norm must be in (0, 1]")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if self.mode == "scheduled_2x2" and (
            self.interval_frames is None or self.interval_frames < 1
        ):
            raise ValueError("scheduled tiling requires a positive interval_frames")
        if self.mode == "uncertainty_2x2" and (
            self.no_confident_small_object_frames is None
            or self.no_confident_small_object_frames < 1
        ):
            raise ValueError(
                "uncertainty tiling requires positive no_confident_small_object_frames"
            )


def detector_runs(frame_index: int, cadence_frames: int) -> bool:
    """Return whether this source frame receives a fresh detector invocation."""
    if frame_index < 0 or cadence_frames < 1:
        raise ValueError("frame_index must be non-negative and cadence positive")
    return frame_index % cadence_frames == 0


def has_confident_small_detection(
    detections: Iterable[TrackingDetection], policy: TilingPolicy
) -> bool:
    """Identify a full-frame result that suppresses uncertainty-triggered tiles."""
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox_norm_xyxy
        if (
            detection.confidence >= policy.confidence_threshold
            and (x2 - x1) * (y2 - y1) <= policy.small_box_max_area_norm
        ):
            return True
    return False


def should_run_tiles(
    policy: TilingPolicy,
    *,
    frame_index: int,
    last_confident_small_detection_frame: int | None,
) -> bool:
    """Decide after a full-frame detection whether a 2x2 scan is required."""
    if policy.mode == "off":
        return False
    if policy.mode == "scheduled_2x2":
        assert policy.interval_frames is not None
        return frame_index % policy.interval_frames == 0
    assert policy.no_confident_small_object_frames is not None
    if last_confident_small_detection_frame is None:
        return frame_index >= policy.no_confident_small_object_frames - 1
    return (
        frame_index - last_confident_small_detection_frame
        >= policy.no_confident_small_object_frames
    )


def tile_boxes_2x2(overlap_fraction: float) -> tuple[Box, Box, Box, Box]:
    """Return overlapping tiles in full-image normalised xyxy coordinates."""
    if not 0.0 <= overlap_fraction < 0.5:
        raise ValueError("tile overlap_fraction must be in [0, 0.5)")
    midpoint_low = 0.5 - overlap_fraction / 2
    midpoint_high = 0.5 + overlap_fraction / 2
    return (
        (0.0, 0.0, midpoint_high, midpoint_high),
        (midpoint_low, 0.0, 1.0, midpoint_high),
        (0.0, midpoint_low, midpoint_high, 1.0),
        (midpoint_low, midpoint_low, 1.0, 1.0),
    )


def tile_detection_to_full_frame(
    detection: TrackingDetection, tile: Box
) -> TrackingDetection:
    """Map a tile-local normalized box back into full-frame coordinates."""
    tx1, ty1, tx2, ty2 = tile
    x1, y1, x2, y2 = detection.bbox_norm_xyxy
    width, height = tx2 - tx1, ty2 - ty1
    return TrackingDetection(
        label=detection.label,
        confidence=detection.confidence,
        bbox_norm_xyxy=(
            tx1 + x1 * width,
            ty1 + y1 * height,
            tx1 + x2 * width,
            ty1 + y2 * height,
        ),
    )


def merge_full_frame_and_tile_detections(
    detections: Iterable[TrackingDetection], *, iou_threshold: float = 0.5
) -> tuple[TrackingDetection, ...]:
    """Class-aware confidence NMS after all boxes are in original coordinates."""
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    kept: list[TrackingDetection] = []
    for candidate in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if all(
            candidate.label != accepted.label
            or _iou(candidate.bbox_norm_xyxy, accepted.bbox_norm_xyxy) < iou_threshold
            for accepted in kept
        ):
            kept.append(candidate)
    return tuple(kept)
