"""IoU matching and PyTorch/ONNX parity summaries."""

from __future__ import annotations

from dataclasses import dataclass

from wam_drones.detection.contracts import Detection, FrameDetections


@dataclass(frozen=True)
class ParityReport:
    matched: int
    reference_only: int
    candidate_only: int
    mean_iou: float
    min_iou: float | None
    max_confidence_delta: float


def bbox_iou(first: Detection, second: Detection) -> float:
    """Compute intersection over union for two normalized xyxy boxes."""
    ax1, ay1, ax2, ay2 = first.bbox_norm_xyxy
    bx1, by1, bx2, by2 = second.bbox_norm_xyxy
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    first_area = (ax2 - ax1) * (ay2 - ay1)
    second_area = (bx2 - bx1) * (by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def compare_frames(
    reference: FrameDetections,
    candidate: FrameDetections,
    *,
    iou_threshold: float = 0.5,
) -> ParityReport:
    """Greedily match same-class boxes by descending IoU."""
    possible = sorted(
        (
            (bbox_iou(left, right), left_index, right_index)
            for left_index, left in enumerate(reference.detections)
            for right_index, right in enumerate(candidate.detections)
            if left.label_id == right.label_id
        ),
        reverse=True,
    )
    used_left: set[int] = set()
    used_right: set[int] = set()
    matches: list[tuple[float, int, int]] = []
    for iou, left_index, right_index in possible:
        if iou < iou_threshold:
            break
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        matches.append((iou, left_index, right_index))
    ious = [match[0] for match in matches]
    confidence_deltas = [
        abs(
            reference.detections[left].confidence
            - candidate.detections[right].confidence
        )
        for _, left, right in matches
    ]
    return ParityReport(
        matched=len(matches),
        reference_only=len(reference.detections) - len(matches),
        candidate_only=len(candidate.detections) - len(matches),
        mean_iou=sum(ious) / len(ious) if ious else 0.0,
        min_iou=min(ious) if ious else None,
        max_confidence_delta=max(confidence_deltas, default=0.0),
    )
