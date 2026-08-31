"""Small, deterministic trackers and leakage-safe VisDrone-MOT scoring.

This deliberately keeps association in the project rather than hiding it behind
the preview-only tracker bundled with a detector package.  It makes every
threshold reviewable, preserves source capture time, and has no ReID path.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import monotonic_ns

import yaml

from wam_drones.dataset.manifest import DatasetManifest
from wam_drones.detection.contracts import (
    FrameDetections,
    FrameTracks,
    TrackObservation,
)
from wam_drones.detection.vocabulary import DETECTION_ID_BY_LABEL, DetectionLabel


@dataclass(frozen=True)
class TrackerConfig:
    """All association thresholds for one reproducible tracker configuration."""

    name: str
    tracker_type: str
    high_confidence_threshold: float
    low_confidence_threshold: float
    new_track_threshold: float
    match_iou_threshold: float
    low_match_iou_threshold: float
    max_time_since_update_frames: int
    min_hits: int
    camera_motion_compensation: bool
    reid_enabled: bool
    source_fps: float = 30.0

    def __post_init__(self) -> None:
        if self.tracker_type not in {"bytetrack", "botsort"}:
            raise ValueError("tracker_type must be bytetrack or botsort")
        for value in (
            self.high_confidence_threshold,
            self.low_confidence_threshold,
            self.new_track_threshold,
            self.match_iou_threshold,
            self.low_match_iou_threshold,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("tracking thresholds must be between zero and one")
        if self.low_confidence_threshold > self.high_confidence_threshold:
            raise ValueError("low confidence threshold cannot exceed high threshold")
        if self.max_time_since_update_frames < 1 or self.min_hits < 1:
            raise ValueError("track lifecycle values must be positive")
        if self.source_fps <= 0:
            raise ValueError("source_fps must be positive")
        if self.reid_enabled:
            raise ValueError("Phase 4 configurations must keep ReID disabled")
        if self.tracker_type == "botsort" and not self.camera_motion_compensation:
            raise ValueError("BoT-SORT requires camera-motion compensation in Phase 4")


def load_tracker_config(path: Path) -> TrackerConfig:
    """Load a strictly named YAML tracker configuration."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"tracker config must be a mapping: {path}")
    allowed = {field.name for field in TrackerConfig.__dataclass_fields__.values()}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"unknown tracker config keys: {sorted(unknown)}")
    return TrackerConfig(**payload)


Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class TrackingDetection:
    """A native-class normalized detector result used for association."""

    label: DetectionLabel
    confidence: float
    bbox_norm_xyxy: Box


@dataclass(frozen=True)
class GroundTruth:
    frame_number: int
    target_id: int
    label: DetectionLabel
    bbox_norm_xyxy: Box
    occlusion: int


@dataclass
class _Track:
    track_id: int
    label: DetectionLabel
    confidence: float
    bbox: Box
    previous_bbox: Box
    age_frames: int = 1
    hits: int = 1
    time_since_update_frames: int = 0

    def velocity(self, frame_interval_s: float) -> tuple[float, float]:
        old_x = (self.previous_bbox[0] + self.previous_bbox[2]) / 2
        old_y = (self.previous_bbox[1] + self.previous_bbox[3]) / 2
        new_x = (self.bbox[0] + self.bbox[2]) / 2
        new_y = (self.bbox[1] + self.bbox[3]) / 2
        return ((new_x - old_x) / frame_interval_s, (new_y - old_y) / frame_interval_s)


def _iou(left: Box, right: Box) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (
        (left[2] - left[0]) * (left[3] - left[1])
        + (right[2] - right[0]) * (right[3] - right[1])
        - intersection
    )
    return intersection / union if union > 0 else 0.0


def _translate(box: Box, dx: float, dy: float) -> Box:
    width, height = box[2] - box[0], box[3] - box[1]
    x1 = min(max(0.0, box[0] + dx), 1.0 - width)
    y1 = min(max(0.0, box[1] + dy), 1.0 - height)
    return (x1, y1, x1 + width, y1 + height)


def _associate(
    tracks: Sequence[_Track], detections: Sequence[TrackingDetection], threshold: float
) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    candidates = sorted(
        (
            (_iou(track.bbox, detection.bbox_norm_xyxy), track_index, detection_index)
            for track_index, track in enumerate(tracks)
            for detection_index, detection in enumerate(detections)
            if track.label == detection.label
        ),
        reverse=True,
    )
    matches: list[tuple[int, int]] = []
    used_tracks: set[int] = set()
    used_detections: set[int] = set()
    for score, track_index, detection_index in candidates:
        if score < threshold:
            break
        if track_index not in used_tracks and detection_index not in used_detections:
            matches.append((track_index, detection_index))
            used_tracks.add(track_index)
            used_detections.add(detection_index)
    return matches, used_tracks, used_detections


class OfflineTracker:
    """ByteTrack two-pass association with optional BoT-SORT camera compensation."""

    def __init__(self, config: TrackerConfig) -> None:
        self.config = config
        self._tracks: list[_Track] = []
        self._next_track_id = 1
        self.last_camera_motion_norm = 0.0

    def update(
        self,
        detections: Iterable[TrackingDetection],
        *,
        frame_id: int,
        captured_at_monotonic_ns: int,
    ) -> FrameTracks:
        """Associate a frame while retaining its capture timestamp verbatim."""
        started = monotonic_ns()
        incoming = tuple(detections)
        self.last_camera_motion_norm = 0.0
        high = tuple(
            d for d in incoming if d.confidence >= self.config.high_confidence_threshold
        )
        low = tuple(
            d
            for d in incoming
            if self.config.low_confidence_threshold
            <= d.confidence
            < self.config.high_confidence_threshold
        )
        for track in self._tracks:
            track.age_frames += 1
            track.time_since_update_frames += 1

        # Estimate global image motion from unambiguous high-confidence matches,
        # then compensate before the actual association pass.
        if self.config.camera_motion_compensation and self._tracks and high:
            provisional, _, _ = _associate(
                self._tracks, high, self.config.match_iou_threshold
            )
            shifts = []
            for track_index, detection_index in provisional:
                track, detection = self._tracks[track_index], high[detection_index]
                tx = (track.bbox[0] + track.bbox[2]) / 2
                ty = (track.bbox[1] + track.bbox[3]) / 2
                dx = (detection.bbox_norm_xyxy[0] + detection.bbox_norm_xyxy[2]) / 2
                dy = (detection.bbox_norm_xyxy[1] + detection.bbox_norm_xyxy[3]) / 2
                shifts.append((dx - tx, dy - ty))
            if shifts:
                offset_x = statistics.median(item[0] for item in shifts)
                offset_y = statistics.median(item[1] for item in shifts)
                self.last_camera_motion_norm = (
                    offset_x * offset_x + offset_y * offset_y
                ) ** 0.5
                for track in self._tracks:
                    track.bbox = _translate(track.bbox, offset_x, offset_y)

        high_matches, matched_track_indexes, matched_high_indexes = _associate(
            self._tracks, high, self.config.match_iou_threshold
        )
        for track_index, detection_index in high_matches:
            self._apply_detection(self._tracks[track_index], high[detection_index])

        unmatched_tracks = [
            track
            for index, track in enumerate(self._tracks)
            if index not in matched_track_indexes
        ]
        low_matches, _, _ = _associate(
            unmatched_tracks, low, self.config.low_match_iou_threshold
        )
        for local_track_index, detection_index in low_matches:
            self._apply_detection(
                unmatched_tracks[local_track_index], low[detection_index]
            )

        for index, detection in enumerate(high):
            if (
                index not in matched_high_indexes
                and detection.confidence >= self.config.new_track_threshold
            ):
                self._tracks.append(self._new_track(detection))
        self._tracks = [
            track
            for track in self._tracks
            if track.time_since_update_frames
            <= self.config.max_time_since_update_frames
        ]
        interval_s = 1.0 / self.config.source_fps
        observations = tuple(
            TrackObservation(
                frame_id=frame_id,
                captured_at_monotonic_ns=captured_at_monotonic_ns,
                track_id=track.track_id,
                label_id=DETECTION_ID_BY_LABEL[track.label],
                label=track.label,
                confidence=track.confidence,
                bbox_norm_xyxy=track.bbox,
                velocity_norm_per_s=track.velocity(interval_s),
                age_frames=track.age_frames,
                hits=track.hits,
                time_since_update_frames=track.time_since_update_frames,
                observed_this_frame=track.time_since_update_frames == 0,
                stale=track.time_since_update_frames >= 3,
            )
            for track in self._tracks
            if track.hits >= self.config.min_hits
        )
        return FrameTracks(
            frame_id=frame_id,
            captured_at_monotonic_ns=captured_at_monotonic_ns,
            produced_at_monotonic_ns=max(
                captured_at_monotonic_ns, monotonic_ns(), started
            ),
            tracks=observations,
        )

    def _new_track(self, detection: TrackingDetection) -> _Track:
        track = _Track(
            self._next_track_id,
            detection.label,
            detection.confidence,
            detection.bbox_norm_xyxy,
            detection.bbox_norm_xyxy,
        )
        self._next_track_id += 1
        return track

    @staticmethod
    def _apply_detection(track: _Track, detection: TrackingDetection) -> None:
        track.previous_bbox = track.bbox
        track.bbox = detection.bbox_norm_xyxy
        track.confidence = detection.confidence
        track.hits += 1
        track.time_since_update_frames = 0


def tracking_detections(frame: FrameDetections) -> tuple[TrackingDetection, ...]:
    return tuple(
        TrackingDetection(item.label, item.confidence, item.bbox_norm_xyxy)
        for item in frame.detections
    )


def _bucket_area(box: Box) -> str:
    area = (box[2] - box[0]) * (box[3] - box[1])
    return "small" if area < 0.0025 else "medium" if area < 0.02 else "large"


def _bucket_occlusion(value: int) -> str:
    return "none" if value == 0 else "partial" if value == 1 else "heavy"


def _bucket_camera_motion(value: float) -> str:
    if value < 0.002:
        return "low"
    if value < 0.01:
        return "medium"
    return "high"


def evaluate_tracking(
    ground_truth: Mapping[int, Sequence[GroundTruth]],
    predictions: Mapping[int, FrameTracks],
    *,
    iou_threshold: float = 0.5,
    camera_motion_by_frame: Mapping[int, float] | None = None,
) -> dict[str, object]:
    """Compute transparent CLEAR/ID-style metrics and HOTA association terms.

    HOTA is reported as sqrt(DetA * AssA), matching the public definition at
    the configured IoU threshold.  The score is intentionally not a substitute
    for the official TrackEval multi-threshold report, but is deterministic and
    suitable for tracker selection within this repository.
    """
    total_gt = matches = false_positive = false_negative = id_switches = fragments = 0
    prior_track_for_gt: dict[int, int] = {}
    was_matched: dict[int, bool] = defaultdict(bool)
    pair_counts: Counter[tuple[int, int]] = Counter()
    gt_counts: Counter[int] = Counter()
    track_counts: Counter[int] = Counter()
    by_bucket: dict[str, Counter[str]] = defaultdict(Counter)
    by_camera_motion: dict[str, Counter[str]] = defaultdict(Counter)
    all_frames = sorted(set(ground_truth) | set(predictions))
    for frame_number in all_frames:
        targets = tuple(ground_truth.get(frame_number, ()))
        tracks = tuple(
            predictions.get(
                frame_number,
                FrameTracks(
                    frame_id=frame_number,
                    captured_at_monotonic_ns=0,
                    produced_at_monotonic_ns=0,
                ),
            ).tracks
        )
        total_gt += len(targets)
        candidates = sorted(
            (
                (
                    _iou(target.bbox_norm_xyxy, track.bbox_norm_xyxy),
                    target_index,
                    track_index,
                )
                for target_index, target in enumerate(targets)
                for track_index, track in enumerate(tracks)
                if target.label == track.label
            ),
            reverse=True,
        )
        used_targets: set[int] = set()
        used_tracks: set[int] = set()
        for score, target_index, track_index in candidates:
            if score < iou_threshold:
                break
            if target_index in used_targets or track_index in used_tracks:
                continue
            target, track = targets[target_index], tracks[track_index]
            used_targets.add(target_index)
            used_tracks.add(track_index)
            matches += 1
            gt_counts[target.target_id] += 1
            track_counts[track.track_id] += 1
            pair_counts[(target.target_id, track.track_id)] += 1
            if (
                target.target_id in prior_track_for_gt
                and prior_track_for_gt[target.target_id] != track.track_id
            ):
                id_switches += 1
            if (
                not was_matched[target.target_id]
                and target.target_id in prior_track_for_gt
            ):
                fragments += 1
            prior_track_for_gt[target.target_id] = track.track_id
            was_matched[target.target_id] = True
            bucket = (
                f"{_bucket_area(target.bbox_norm_xyxy)}:"
                f"{_bucket_occlusion(target.occlusion)}"
            )
            by_bucket[bucket]["matched"] += 1
            by_bucket[bucket]["ground_truth"] += 1
            motion = (camera_motion_by_frame or {}).get(frame_number, 0.0)
            motion_bucket = _bucket_camera_motion(motion)
            by_camera_motion[motion_bucket]["matched"] += 1
            by_camera_motion[motion_bucket]["ground_truth"] += 1
        for index, target in enumerate(targets):
            if index not in used_targets:
                false_negative += 1
                was_matched[target.target_id] = False
                bucket = (
                    f"{_bucket_area(target.bbox_norm_xyxy)}:"
                    f"{_bucket_occlusion(target.occlusion)}"
                )
                by_bucket[bucket]["ground_truth"] += 1
                motion = (camera_motion_by_frame or {}).get(frame_number, 0.0)
                motion_bucket = _bucket_camera_motion(motion)
                by_camera_motion[motion_bucket]["ground_truth"] += 1
        false_positive += len(tracks) - len(used_tracks)
    association_numerator = sum(count * count for count in pair_counts.values())
    association_denominator = sum(
        count * (gt_counts[gt_id] + track_counts[track_id] - count)
        for (gt_id, track_id), count in pair_counts.items()
    )
    deta = (
        matches / (matches + false_positive + false_negative)
        if (matches + false_positive + false_negative)
        else 0.0
    )
    assa = (
        association_numerator / association_denominator
        if association_denominator
        else 0.0
    )
    idtp = sum(
        max(
            (count for (gt_id, _), count in pair_counts.items() if gt_id == target_id),
            default=0,
        )
        for target_id in gt_counts
    )
    idfp = sum(track_counts.values()) - idtp
    idfn = sum(gt_counts.values()) - idtp
    breakdowns = {
        bucket: {
            "track_recall": values["matched"] / values["ground_truth"]
            if values["ground_truth"]
            else 0.0,
            **dict(values),
        }
        for bucket, values in sorted(by_bucket.items())
    }
    camera_breakdowns = {
        bucket: {
            "track_recall": values["matched"] / values["ground_truth"]
            if values["ground_truth"]
            else 0.0,
            **dict(values),
        }
        for bucket, values in sorted(by_camera_motion.items())
    }
    return {
        "hota": (deta * assa) ** 0.5,
        "deta": deta,
        "assa": assa,
        "idf1": 2 * idtp / (2 * idtp + idfp + idfn) if 2 * idtp + idfp + idfn else 0.0,
        "identity_switches": id_switches,
        "fragmentation": fragments,
        "track_recall": matches / total_gt if total_gt else 0.0,
        "matches": matches,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "ground_truth": total_gt,
        "breakdowns": breakdowns,
        "camera_motion_breakdowns": camera_breakdowns,
    }


def load_visdrone_ground_truth(
    manifest: DatasetManifest, repo_root: Path
) -> dict[str, dict[int, tuple[GroundTruth, ...]]]:
    """Load tracked native annotations, including identity and occlusion metadata."""
    output: dict[str, dict[int, tuple[GroundTruth, ...]]] = defaultdict(dict)
    for sample in manifest.samples:
        if (
            sample.sequence_id is None
            or sample.frame_number is None
            or sample.relative_annotation_path is None
        ):
            raise ValueError(
                "MOT samples require sequence, frame number, and annotations"
            )
        payload = json.loads(
            (repo_root / sample.relative_annotation_path).read_text(encoding="utf-8")
        )
        rows = []
        for box in payload["boxes"]:
            if not box["is_trainable"]:
                continue
            x1, y1, x2, y2 = (float(value) for value in box["bbox_xyxy_px"])
            rows.append(
                GroundTruth(
                    sample.frame_number,
                    int(box["target_id"]),
                    DetectionLabel(
                        list(DETECTION_ID_BY_LABEL)[int(box["category"]) - 1]
                    ),
                    (
                        x1 / sample.width_px,
                        y1 / sample.height_px,
                        x2 / sample.width_px,
                        y2 / sample.height_px,
                    ),
                    int(box["occlusion"]),
                )
            )
        output[sample.sequence_id][sample.frame_number] = tuple(rows)
    return dict(output)


def mot_lines(
    frames: Iterable[FrameTracks], width_px: int, height_px: int
) -> list[str]:
    """Serialize tracks in MOTChallenge order without changing frame timestamps."""
    lines = []
    for frame in frames:
        for track in frame.tracks:
            x1, y1, x2, y2 = track.bbox_norm_xyxy
            values = (
                frame.frame_id,
                track.track_id,
                f"{x1 * width_px:.3f}",
                f"{y1 * height_px:.3f}",
                f"{(x2 - x1) * width_px:.3f}",
                f"{(y2 - y1) * height_px:.3f}",
                f"{track.confidence:.6f}",
                track.label_id,
                -1,
                -1,
            )
            lines.append(",".join(str(value) for value in values))
    return lines
