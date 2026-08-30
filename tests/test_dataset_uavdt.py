from pathlib import Path

import pytest
from PIL import Image

from wam_drones.dataset.uavdt import (
    convert_uavdt_split,
    parse_uavdt_gt_ignore,
    parse_uavdt_gt_whole,
)

GT_WHOLE_TEXT = """\
1,1,100,100,20,40,0,0,1
1,2,300,300,40,20,0,1,2
2,1,105,102,20,40,0,0,1
"""

GT_IGNORE_TEXT = """\
1,0,0,50,50
"""


def test_parse_uavdt_gt_whole_reads_all_fields() -> None:
    boxes = parse_uavdt_gt_whole(GT_WHOLE_TEXT)

    assert len(boxes) == 3
    car, truck, _ = boxes
    assert car.category == 1
    assert car.label is not None
    assert car.label.value == "car"
    assert truck.category == 2
    assert truck.label is not None
    assert truck.label.value == "truck"
    assert car.bbox_xyxy_px == (100.0, 100.0, 120.0, 140.0)


def test_parse_uavdt_gt_whole_rejects_malformed_row() -> None:
    with pytest.raises(ValueError, match="expected 9 fields"):
        parse_uavdt_gt_whole("1,1,100,100,20,40,0,0\n")


def test_parse_uavdt_gt_ignore_reads_regions() -> None:
    regions = parse_uavdt_gt_ignore(GT_IGNORE_TEXT)

    assert len(regions) == 1
    assert regions[0].frame_index == 1
    assert regions[0].bbox_xyxy_px == (0.0, 0.0, 50.0, 50.0)


def test_parse_uavdt_gt_ignore_rejects_malformed_row() -> None:
    with pytest.raises(ValueError, match="expected 5 fields"):
        parse_uavdt_gt_ignore("1,0,0,50\n")


def make_source_split(tmp_path: Path) -> Path:
    source_dir = tmp_path / "source" / "UAVDT"
    sequence_dir = source_dir / "UAV-benchmark-M" / "M0101"
    sequence_dir.mkdir(parents=True)
    (source_dir / "GT").mkdir(parents=True)
    Image.new("RGB", (400, 300), color="green").save(sequence_dir / "img000001.jpg")
    Image.new("RGB", (400, 300), color="green").save(sequence_dir / "img000002.jpg")
    (source_dir / "GT" / "M0101_gt_whole.txt").write_text(
        GT_WHOLE_TEXT, encoding="utf-8"
    )
    (source_dir / "GT" / "M0101_gt_ignore.txt").write_text(
        GT_IGNORE_TEXT, encoding="utf-8"
    )
    return source_dir


def test_convert_uavdt_split_builds_per_frame_manifest(tmp_path: Path) -> None:
    repo_root = tmp_path
    source_dir = make_source_split(tmp_path)

    manifest = convert_uavdt_split(
        source_dir,
        "train",
        repo_root / "data" / "raw" / "uavdt",
        repo_root,
        "manual-download://uavdt-train",
    )

    assert manifest.dataset == "uavdt"
    assert len(manifest.samples) == 2
    by_frame = {sample.frame_number: sample for sample in manifest.samples}
    assert by_frame[1].box_count == 2
    assert by_frame[1].ignored_region_count == 1
    assert by_frame[2].box_count == 1
    assert by_frame[2].ignored_region_count == 0
    for sample in manifest.samples:
        assert sample.relative_label_path is not None
        label_lines = (repo_root / sample.relative_label_path).read_text(
            encoding="utf-8"
        )
        assert label_lines.strip() != "" or sample.box_count == 0


def test_convert_uavdt_split_rejects_empty_sequences(tmp_path: Path) -> None:
    source_dir = tmp_path / "source" / "UAVDT"
    (source_dir / "UAV-benchmark-M").mkdir(parents=True)
    (source_dir / "GT").mkdir(parents=True)

    with pytest.raises(ValueError, match="no sequences found"):
        convert_uavdt_split(
            source_dir, "train", tmp_path / "out", tmp_path, "https://example.invalid"
        )
