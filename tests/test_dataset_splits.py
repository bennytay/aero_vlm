from datetime import UTC, datetime

import pytest

from wam_drones.dataset.manifest import DatasetManifest, ImageRecord
from wam_drones.dataset.splits import validate_split_integrity


def make_record(**overrides: object) -> ImageRecord:
    fields: dict[str, object] = {
        "dataset": "visdrone_mot",
        "split": "train",
        "source_url": "manual-download://visdrone-mot",
        "original_id": "seq_0000001",
        "relative_image_path": "a.jpg",
        "sha256": "sha256:" + "c" * 64,
        "width_px": 100,
        "height_px": 100,
        "sequence_id": "seq",
        "frame_number": 1,
        "annotation_conversion_version": "v1",
        "box_count": 0,
        "ignored_region_count": 0,
    }
    fields.update(overrides)
    return ImageRecord.model_validate(fields)


def make_manifest(dataset: str, split: str, **overrides: object) -> DatasetManifest:
    return DatasetManifest(
        dataset=dataset,  # type: ignore[arg-type]
        split=split,  # type: ignore[arg-type]
        created_utc=datetime(2026, 8, 31, tzinfo=UTC),
        samples=(make_record(dataset=dataset, split=split, **overrides),),
    )


def test_validate_split_integrity_passes_for_disjoint_sequences_and_hashes() -> None:
    train = make_manifest(
        "visdrone_mot",
        "train",
        sequence_id="seq-train",
        sha256="sha256:" + "1" * 64,
        original_id="train_0000001",
    )
    val = make_manifest(
        "visdrone_mot",
        "val",
        sequence_id="seq-val",
        sha256="sha256:" + "2" * 64,
        original_id="val_0000001",
    )

    validate_split_integrity([train, val])


def test_validate_split_integrity_rejects_sequence_spanning_splits() -> None:
    train = make_manifest(
        "visdrone_mot",
        "train",
        sequence_id="shared-seq",
        sha256="sha256:" + "1" * 64,
        original_id="train_0000001",
    )
    val = make_manifest(
        "visdrone_mot",
        "val",
        sequence_id="shared-seq",
        sha256="sha256:" + "2" * 64,
        original_id="val_0000001",
    )

    with pytest.raises(ValueError, match="sequence shared-seq appears in both"):
        validate_split_integrity([train, val])


def test_validate_split_integrity_rejects_duplicate_hash_across_splits() -> None:
    shared_sha = "sha256:" + "3" * 64
    train = make_manifest(
        "visdrone_det",
        "train",
        sequence_id=None,
        frame_number=None,
        sha256=shared_sha,
        original_id="train_a",
    )
    val = make_manifest(
        "visdrone_det",
        "val",
        sequence_id=None,
        frame_number=None,
        sha256=shared_sha,
        original_id="val_a",
    )

    with pytest.raises(ValueError, match="appears in both"):
        validate_split_integrity([train, val])
