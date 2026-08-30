import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from wam_drones.detection import (
    Detection,
    FrameDetections,
    FrameTracks,
    TrackObservation,
    validate_frame_sequence,
)


def make_detection(**changes: object) -> Detection:
    values: dict[str, object] = {
        "frame_id": 7,
        "captured_at_monotonic_ns": 1_000_000,
        "label_id": 4,
        "label": "car",
        "confidence": 0.91,
        "bbox_norm_xyxy": (0.1, 0.2, 0.4, 0.6),
        "model_name": "detector-v1",
    }
    values.update(changes)
    return Detection.model_validate(values)


def make_detection_frame(**changes: object) -> FrameDetections:
    values: dict[str, object] = {
        "frame_id": 7,
        "captured_at_monotonic_ns": 1_000_000,
        "inference_started_at_monotonic_ns": 1_100_000,
        "produced_at_monotonic_ns": 1_500_000,
        "image_width_px": 1920,
        "image_height_px": 1080,
        "detections": (make_detection(),),
    }
    values.update(changes)
    return FrameDetections.model_validate(values)


def make_track(**changes: object) -> TrackObservation:
    values: dict[str, object] = {
        "frame_id": 7,
        "captured_at_monotonic_ns": 1_000_000,
        "track_id": 12,
        "label_id": 4,
        "label": "car",
        "confidence": 0.88,
        "bbox_norm_xyxy": (0.11, 0.2, 0.41, 0.6),
        "velocity_norm_per_s": (0.02, -0.01),
        "age_frames": 5,
        "hits": 4,
        "time_since_update_frames": 0,
        "observed_this_frame": True,
        "stale": False,
    }
    values.update(changes)
    return TrackObservation.model_validate(values)


def test_contracts_roundtrip_through_json() -> None:
    detection = make_detection()
    track = make_track()

    assert Detection.model_validate_json(detection.model_dump_json()) == detection
    assert TrackObservation.model_validate_json(track.model_dump_json()) == track


@pytest.mark.parametrize(
    "box",
    [
        (0.4, 0.2, 0.1, 0.6),
        (0.1, 0.6, 0.4, 0.2),
        (0.1, 0.2, 0.1, 0.6),
        (0.1, 0.2, 0.4, 0.2),
    ],
)
def test_detection_rejects_inverted_or_zero_area_boxes(
    box: tuple[float, float, float, float],
) -> None:
    with pytest.raises(ValidationError, match="positive width and height"):
        make_detection(bbox_norm_xyxy=box)


def test_detection_rejects_unknown_or_mismatched_labels() -> None:
    with pytest.raises(ValidationError):
        make_detection(label="helicopter")
    with pytest.raises(ValidationError, match="must match"):
        make_detection(label_id=3, label="car")


def test_frame_rejects_non_monotonic_processing_time() -> None:
    with pytest.raises(ValidationError, match="timestamps must be monotonic"):
        make_detection_frame(inference_started_at_monotonic_ns=900_000)


def test_frame_rejects_detection_from_another_frame() -> None:
    with pytest.raises(ValidationError, match="match the frame_id"):
        make_detection_frame(detections=(make_detection(frame_id=8),))


def test_track_lifecycle_and_staleness_are_consistent() -> None:
    predicted = make_track(
        time_since_update_frames=3,
        observed_this_frame=False,
        stale=True,
    )
    assert predicted.stale is True

    with pytest.raises(ValidationError, match="observed_this_frame"):
        make_track(time_since_update_frames=1, observed_this_frame=True)
    with pytest.raises(ValidationError, match="stale"):
        make_track(
            time_since_update_frames=3,
            observed_this_frame=False,
            stale=False,
        )


def test_frame_tracks_rejects_duplicate_track_ids() -> None:
    track = make_track()
    with pytest.raises(ValidationError, match="unique"):
        FrameTracks(
            frame_id=7,
            captured_at_monotonic_ns=1_000_000,
            produced_at_monotonic_ns=1_500_000,
            tracks=(track, track),
        )


def test_frame_sequence_requires_increasing_ids_and_capture_times() -> None:
    first = make_detection_frame()
    second = make_detection_frame(
        frame_id=8,
        captured_at_monotonic_ns=2_000_000,
        inference_started_at_monotonic_ns=2_100_000,
        produced_at_monotonic_ns=2_500_000,
        detections=(make_detection(frame_id=8, captured_at_monotonic_ns=2_000_000),),
    )
    validate_frame_sequence((first, second))

    backwards_time = make_detection_frame(
        frame_id=8,
        captured_at_monotonic_ns=900_000,
        inference_started_at_monotonic_ns=950_000,
        produced_at_monotonic_ns=990_000,
        detections=(make_detection(frame_id=8, captured_at_monotonic_ns=900_000),),
    )
    with pytest.raises(ValueError, match="capture times"):
        validate_frame_sequence((first, backwards_time))
    with pytest.raises(ValueError, match="frame IDs"):
        validate_frame_sequence((second, first))


def test_checked_in_schemas_match_models() -> None:
    schema_dir = Path(__file__).parents[1] / "interfaces" / "schemas"
    expected = {
        "detection_v1.schema.json": Detection.model_json_schema(),
        "frame_detections_v1.schema.json": FrameDetections.model_json_schema(),
        "frame_tracks_v1.schema.json": FrameTracks.model_json_schema(),
        "track_observation_v1.schema.json": TrackObservation.model_json_schema(),
    }
    for filename, schema in expected.items():
        checked_in = json.loads((schema_dir / filename).read_text(encoding="utf-8"))
        assert checked_in == schema
