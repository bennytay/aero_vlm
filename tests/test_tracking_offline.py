from wam_drones.detection.vocabulary import DetectionLabel
from wam_drones.tracking.offline import (
    GroundTruth,
    OfflineTracker,
    TrackerConfig,
    TrackingDetection,
    evaluate_tracking,
)


def config() -> TrackerConfig:
    return TrackerConfig(
        name="test",
        tracker_type="bytetrack",
        high_confidence_threshold=0.5,
        low_confidence_threshold=0.1,
        new_track_threshold=0.5,
        match_iou_threshold=0.3,
        low_match_iou_threshold=0.2,
        max_time_since_update_frames=4,
        min_hits=1,
        camera_motion_compensation=False,
        reid_enabled=False,
    )


def test_bytetrack_keeps_id_through_low_score_detection_and_capture_time() -> None:
    tracker = OfflineTracker(config())
    first = tracker.update(
        (TrackingDetection(DetectionLabel.CAR, 0.9, (0.1, 0.1, 0.2, 0.2)),),
        frame_id=10,
        captured_at_monotonic_ns=1_000,
    )
    second = tracker.update(
        (TrackingDetection(DetectionLabel.CAR, 0.2, (0.11, 0.1, 0.21, 0.2)),),
        frame_id=11,
        captured_at_monotonic_ns=2_000,
    )

    assert first.tracks[0].track_id == second.tracks[0].track_id
    assert second.tracks[0].observed_this_frame is True
    assert second.captured_at_monotonic_ns == 2_000


def test_metrics_report_identity_switches_fragmentation_and_breakdowns() -> None:
    tracker = OfflineTracker(config())
    frame_one = tracker.update(
        (TrackingDetection(DetectionLabel.CAR, 0.9, (0.1, 0.1, 0.2, 0.2)),),
        frame_id=0,
        captured_at_monotonic_ns=1,
    )
    # A distinct tracker instance simulates an identity reset on the next frame.
    reset_tracker = OfflineTracker(config())
    frame_two = reset_tracker.update(
        (TrackingDetection(DetectionLabel.CAR, 0.9, (0.1, 0.1, 0.2, 0.2)),),
        frame_id=1,
        captured_at_monotonic_ns=2,
    )
    ground_truth = {
        0: (GroundTruth(0, 7, DetectionLabel.CAR, (0.1, 0.1, 0.2, 0.2), 0),),
        1: (GroundTruth(1, 7, DetectionLabel.CAR, (0.1, 0.1, 0.2, 0.2), 2),),
    }
    # Change the second ID to make the reset observable to the scorer.
    changed = frame_two.model_copy(
        update={"tracks": (frame_two.tracks[0].model_copy(update={"track_id": 2}),)}
    )
    metrics = evaluate_tracking(ground_truth, {0: frame_one, 1: changed})

    assert metrics["track_recall"] == 1.0
    assert metrics["identity_switches"] == 1
    assert set(metrics["breakdowns"]) == {"medium:none", "medium:heavy"}
