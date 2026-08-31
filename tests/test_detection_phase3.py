import json
from datetime import UTC, datetime
from pathlib import Path

from wam_drones.dataset.manifest import DatasetManifest, ImageRecord
from wam_drones.detection.phase3 import (
    Box,
    ImageEvaluationInput,
    evaluate_predictions,
    prepare_phase3_dataset,
)


def sample(original_id: str, split: str) -> ImageRecord:
    return ImageRecord(
        dataset="visdrone_det",
        split=split,  # type: ignore[arg-type]
        source_url="https://example.test/visdrone",
        original_id=original_id,
        relative_image_path=f"data/raw/visdrone_det/{split}/images/{original_id}.jpg",
        sha256="sha256:" + "0" * 64,
        width_px=1280,
        height_px=720,
        annotation_conversion_version="test_v1",
        relative_label_path=f"data/raw/visdrone_det/{split}/labels/{original_id}.txt",
        relative_annotation_path=(
            f"data/raw/visdrone_det/{split}/annotations_full/{original_id}.json"
        ),
        box_count=1,
        ignored_region_count=0,
    )


def manifest(split: str, samples: tuple[ImageRecord, ...]) -> DatasetManifest:
    return DatasetManifest(
        dataset="visdrone_det",
        split=split,  # type: ignore[arg-type]
        created_utc=datetime(2026, 8, 31, tzinfo=UTC),
        samples=samples,
    )


def test_prepare_excludes_only_known_validation_leak(tmp_path: Path) -> None:
    train = manifest("train", (sample("0000323_train", "train"),))
    val = manifest(
        "val", (sample("0000323_val", "val"), sample("clean_val", "val"))
    )
    train_path = tmp_path / "train.json"
    val_path = tmp_path / "val.json"
    train.write(train_path)
    val.write(val_path)
    dedup = tmp_path / "dedup.json"
    dedup.write_text(
        json.dumps(
            {"groups": [{"original_ids": ["0000323_train", "0000323_val"]}]}
        ),
        encoding="utf-8",
    )

    result = prepare_phase3_dataset(
        repo_root=tmp_path,
        output_dir=tmp_path / "run",
        train_manifest_path=train_path,
        val_manifest_path=val_path,
        dedup_report_path=dedup,
    )

    assert result["excluded_validation_original_ids"] == ["0000323_val"]
    assert (tmp_path / "run" / "sequence_safe_val_images.txt").read_text(
        encoding="utf-8"
    ).endswith("clean_val.jpg\n")
    safe = DatasetManifest.read(tmp_path / "run" / "sequence_safe_val_manifest.json")
    assert [item.original_id for item in safe.samples] == ["clean_val"]


def test_metrics_include_native_class_small_and_all_breakdowns() -> None:
    image = ImageEvaluationInput(
        sample("0000323_eval", "val"),
        targets=(Box(0, (0.0, 0.0, 10.0, 10.0), occlusion=2),),
        ignored_regions=(),
    )
    result = evaluate_predictions(
        (image,), {"0000323_eval": (Box(0, (0.0, 0.0, 10.0, 10.0), 0.9),)}
    )

    person = result["per_class"]["pedestrian"]
    assert person["map50_95"] == 1.0
    assert person["ap_small"] == 1.0
    assert person["precision"] == 1.0
    assert person["recall"] == 1.0
    assert person["false_positives_per_frame"] == 0.0
    assert set(result["breakdowns"]) == {"box_size", "density", "occlusion", "scene"}
    assert result["breakdowns"]["occlusion"]["heavy"]["ground_truth_count"] == 1
