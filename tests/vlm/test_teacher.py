import json
from pathlib import Path

import pytest

from wam_drones.detection.contracts import FrameTracks, TrackObservation
from wam_drones.detection.vocabulary import DetectionLabel
from wam_drones.vlm.teacher import (
    FrameProvenance,
    TeacherRunMetadata,
    load_teacher_artifact,
    validate_track_jsonl,
    write_provenance_index,
    write_teacher_metadata,
)


def _frame(frame_id: int, captured_ns: int) -> FrameTracks:
    return FrameTracks(
        frame_id=frame_id,
        captured_at_monotonic_ns=captured_ns,
        produced_at_monotonic_ns=captured_ns,
        tracks=(
            TrackObservation(
                frame_id=frame_id,
                captured_at_monotonic_ns=captured_ns,
                track_id=9,
                label_id=4,
                label=DetectionLabel.CAR,
                confidence=0.8,
                bbox_norm_xyxy=(0.1, 0.2, 0.3, 0.4),
                velocity_norm_per_s=(0.0, 0.0),
                age_frames=2,
                hits=2,
                time_since_update_frames=0,
                observed_this_frame=True,
                stale=False,
            ),
        ),
    )


def _metadata() -> TeacherRunMetadata:
    return TeacherRunMetadata(
        artifact_kind="evaluation",
        detector_checkpoint="models/best.pt",
        detector_checkpoint_sha256="a" * 64,
        tracker_config="configs/tracking/botsort_phase4.yaml",
        tracker_config_sha256="b" * 64,
        tracker_name="botsort_phase4_no_reid_cmc",
        tracker_thresholds={"min_hits": 1, "reid_enabled": False},
        protocol="configs/tracking/phase4_protocol.yaml",
        protocol_sha256="c" * 64,
        code_revision="deadbeef",
    )


def test_teacher_reader_round_trips_tracks_and_source_identity(tmp_path: Path) -> None:
    tracks_path = tmp_path / "tracks.jsonl"
    provenance_path = tmp_path / "frame_provenance.jsonl"
    metadata_path = tmp_path / "teacher_metadata.json"
    frames = (_frame(0, 0), _frame(1, 10))
    tracks_path.write_text("".join(item.model_dump_json() + "\n" for item in frames))
    write_provenance_index(
        provenance_path,
        [
            FrameProvenance(
                frame_id=0,
                sequence_id="sequence-a",
                source_frame_number=1,
                source_path="sequences/sequence-a/0000001.jpg",
                image_width_px=1920,
                image_height_px=1080,
                source_sha256="d" * 64,
            ),
            FrameProvenance(
                frame_id=1,
                sequence_id="sequence-a",
                source_frame_number=2,
                source_path="sequences/sequence-a/0000002.jpg",
                image_width_px=1920,
                image_height_px=1080,
                source_sha256="e" * 64,
            ),
        ],
    )
    write_teacher_metadata(metadata_path, _metadata())

    artifact = load_teacher_artifact(tracks_path, provenance_path, metadata_path)

    provenance, tracks = next(artifact.observations())
    assert provenance.source_frame_number == 1
    assert tracks.tracks[0].track_id == 9
    assert tracks.tracks[0].observed_this_frame is True


def test_teacher_reader_rejects_missing_source_mapping(tmp_path: Path) -> None:
    tracks_path = tmp_path / "tracks.jsonl"
    provenance_path = tmp_path / "frame_provenance.jsonl"
    metadata_path = tmp_path / "teacher_metadata.json"
    tracks_path.write_text(_frame(0, 0).model_dump_json() + "\n")
    provenance_path.write_text("")
    write_teacher_metadata(metadata_path, _metadata())

    with pytest.raises(ValueError, match="one-to-one"):
        load_teacher_artifact(tracks_path, provenance_path, metadata_path)


def test_demo_sidecar_validation_rejects_invalid_json(tmp_path: Path) -> None:
    tracks_path = tmp_path / "demo.tracks.jsonl"
    tracks_path.write_text(json.dumps({"frame_id": 0}) + "\n")

    with pytest.raises(ValueError, match=r"invalid demo\.tracks\.jsonl record"):
        validate_track_jsonl(tracks_path)
