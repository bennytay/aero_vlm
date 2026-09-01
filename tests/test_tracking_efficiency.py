from pathlib import Path

import pytest

from wam_drones.detection.vocabulary import DetectionLabel
from wam_drones.tracking.efficiency import (
    TilingPolicy,
    detector_runs,
    load_efficiency_protocol,
    merge_full_frame_and_tile_detections,
    should_run_tiles,
    tile_boxes_2x2,
    tile_detection_to_full_frame,
    validate_efficiency_profile,
)
from wam_drones.tracking.offline import TrackingDetection


def test_cadence_runs_fresh_detection_only_on_configured_frames() -> None:
    assert [detector_runs(index, 3) for index in range(6)] == [
        True,
        False,
        False,
        True,
        False,
        False,
    ]


def test_uncertainty_tiling_waits_for_configured_absence_interval() -> None:
    policy = TilingPolicy("uncertainty_2x2", no_confident_small_object_frames=3)
    assert not should_run_tiles(
        policy, frame_index=1, last_confident_small_detection_frame=None
    )
    assert should_run_tiles(
        policy, frame_index=2, last_confident_small_detection_frame=None
    )
    assert not should_run_tiles(
        policy, frame_index=7, last_confident_small_detection_frame=5
    )
    assert should_run_tiles(
        policy, frame_index=8, last_confident_small_detection_frame=5
    )


def test_tile_coordinates_are_returned_to_original_frame_before_merging() -> None:
    tile = tile_boxes_2x2(0.2)[3]
    local = TrackingDetection(DetectionLabel.CAR, 0.8, (0.0, 0.0, 0.5, 0.5))
    full = tile_detection_to_full_frame(local, tile)

    assert full.bbox_norm_xyxy == pytest.approx((0.4, 0.4, 0.7, 0.7))
    duplicate = TrackingDetection(DetectionLabel.CAR, 0.9, (0.4, 0.4, 0.7, 0.7))
    assert merge_full_frame_and_tile_detections((full, duplicate)) == (duplicate,)


def test_protocol_only_accepts_pre_registered_profile() -> None:
    protocol = load_efficiency_protocol(
        Path("configs/experiments/phase5_efficiency.yaml")
    )
    policy = next(item for item in protocol.tiling_policies if item.mode == "off")
    validate_efficiency_profile(
        protocol,
        input_size_px=640,
        precision="fp32",
        detector_cadence_frames=1,
        tiling_policy=policy,
        tracker_config="configs/tracking/bytetrack_phase4.yaml",
        preview=False,
    )
    with pytest.raises(ValueError, match="input size"):
        validate_efficiency_profile(
            protocol,
            input_size_px=384,
            precision="fp32",
            detector_cadence_frames=1,
            tiling_policy=policy,
            tracker_config="configs/tracking/bytetrack_phase4.yaml",
            preview=False,
        )
