from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from wam_drones.dataset.manifest import DatasetManifest, ImageRecord

VALID_SHA = "sha256:" + "a" * 64


def make_record(**overrides: object) -> ImageRecord:
    fields: dict[str, object] = {
        "dataset": "visdrone_det",
        "split": "train",
        "source_url": "https://example.invalid/archive.zip",
        "original_id": "0000001_00000_d_0000001",
        "relative_image_path": "data/raw/visdrone_det/train/images/a.jpg",
        "sha256": VALID_SHA,
        "width_px": 1920,
        "height_px": 1080,
        "annotation_conversion_version": "v1",
        "relative_label_path": "data/raw/visdrone_det/train/labels/a.txt",
        "relative_annotation_path": (
            "data/raw/visdrone_det/train/annotations_full/a.json"
        ),
        "box_count": 5,
        "ignored_region_count": 1,
    }
    fields.update(overrides)
    return ImageRecord.model_validate(fields)


def test_manifest_round_trips_through_json(tmp_path: Path) -> None:
    manifest = DatasetManifest(
        dataset="visdrone_det",
        split="train",
        created_utc=datetime(2026, 8, 31, tzinfo=UTC),
        samples=(make_record(),),
    )
    path = tmp_path / "manifest.json"
    manifest.write(path)

    loaded = DatasetManifest.read(path)

    assert loaded == manifest


def test_manifest_rejects_sample_dataset_mismatch() -> None:
    with pytest.raises(ValidationError, match="must match the manifest dataset"):
        DatasetManifest(
            dataset="visdrone_det",
            split="train",
            created_utc=datetime(2026, 8, 31, tzinfo=UTC),
            samples=(make_record(dataset="uavdt"),),
        )


def test_manifest_rejects_duplicate_original_id() -> None:
    with pytest.raises(ValidationError, match="duplicate original_id"):
        DatasetManifest(
            dataset="visdrone_det",
            split="train",
            created_utc=datetime(2026, 8, 31, tzinfo=UTC),
            samples=(
                make_record(relative_image_path="a.jpg"),
                make_record(relative_image_path="b.jpg"),
            ),
        )


def test_image_record_rejects_bad_sha256_shape() -> None:
    with pytest.raises(ValidationError):
        make_record(sha256="not-a-hash")


def test_image_record_requires_sequence_and_frame_together() -> None:
    with pytest.raises(ValidationError, match="must both be set or both be absent"):
        make_record(sequence_id="seq-1", frame_number=None)
    with pytest.raises(ValidationError, match="must both be set or both be absent"):
        make_record(sequence_id=None, frame_number=3)


def test_image_record_allows_sequence_and_frame_together() -> None:
    record = make_record(sequence_id="uav0000086_00000_v", frame_number=12)
    assert record.sequence_id == "uav0000086_00000_v"
    assert record.frame_number == 12
