import pytest

from wam_drones.detection.contracts import Detection, FrameDetections
from wam_drones.detection.parity import bbox_iou, compare_frames


def detection(
    box: tuple[float, float, float, float],
    *,
    label_id: int = 4,
    label: str = "car",
    confidence: float = 0.9,
) -> Detection:
    return Detection.model_validate(
        {
            "frame_id": 0,
            "captured_at_monotonic_ns": 10,
            "label_id": label_id,
            "label": label,
            "confidence": confidence,
            "bbox_norm_xyxy": box,
            "model_name": "test",
        }
    )


def frame(*detections: Detection) -> FrameDetections:
    return FrameDetections(
        frame_id=0,
        captured_at_monotonic_ns=10,
        inference_started_at_monotonic_ns=11,
        produced_at_monotonic_ns=12,
        image_width_px=100,
        image_height_px=100,
        detections=detections,
    )


def test_iou_and_greedy_same_class_matching() -> None:
    exact = detection((0.1, 0.1, 0.4, 0.4))
    shifted = detection((0.2, 0.1, 0.5, 0.4), confidence=0.8)
    bus = detection((0.1, 0.1, 0.4, 0.4), label_id=9, label="bus")

    assert bbox_iou(exact, exact) == 1.0
    report = compare_frames(frame(exact, bus), frame(shifted), iou_threshold=0.49)
    assert report.matched == 1
    assert report.reference_only == 1
    assert report.candidate_only == 0
    assert report.mean_iou == pytest.approx(0.5)
    assert report.max_confidence_delta == pytest.approx(0.1)
