import json
from pathlib import Path

import pytest
from PIL import Image

from wam_drones.dataset.visdrone_det import (
    convert_visdrone_det_split,
    parse_visdrone_det_annotation,
)

ANNOTATION_TEXT = """\
0,0,50,50,1,0,0,0
100,100,20,40,1,1,1,2
200,200,30,30,1,11,0,0
"""


def test_parse_visdrone_det_annotation_reads_all_fields() -> None:
    boxes = parse_visdrone_det_annotation(ANNOTATION_TEXT)

    assert len(boxes) == 3
    ignored, pedestrian, others = boxes
    assert ignored.category == 0
    assert ignored.is_ignored_region
    assert not ignored.is_trainable
    assert pedestrian.category == 1
    assert pedestrian.truncation == 1
    assert pedestrian.occlusion == 2
    assert pedestrian.is_trainable
    assert pedestrian.bbox_xyxy_px == (100.0, 100.0, 120.0, 140.0)
    assert others.category == 11
    assert not others.is_trainable
    assert not others.is_ignored_region


def test_parse_visdrone_det_annotation_rejects_malformed_row() -> None:
    with pytest.raises(ValueError, match="expected 8 fields"):
        parse_visdrone_det_annotation("0,0,50,50,1,0,0\n")


def test_parse_visdrone_det_annotation_rejects_negative_size() -> None:
    with pytest.raises(ValueError, match="negative box size"):
        parse_visdrone_det_annotation("0,0,-1,50,1,1,0,0\n")


def test_parse_visdrone_det_annotation_allows_zero_size_ignore_regions() -> None:
    # Real VisDrone GT ships degenerate zero-height ignore-region rows
    # (e.g. category 0, height 0); these must parse, not crash the pipeline.
    boxes = parse_visdrone_det_annotation("1008,374,3,0,0,0,0,0\n")

    assert len(boxes) == 1
    assert boxes[0].is_ignored_region
    assert boxes[0].bbox_xyxy_px == (1008.0, 374.0, 1011.0, 374.0)


def make_source_split(tmp_path: Path) -> Path:
    source_dir = tmp_path / "source" / "VisDrone2019-DET-val"
    (source_dir / "images").mkdir(parents=True)
    (source_dir / "annotations").mkdir(parents=True)
    Image.new("RGB", (400, 300), color="blue").save(
        source_dir / "images" / "0000001_00000_d_0000001.jpg"
    )
    (source_dir / "annotations" / "0000001_00000_d_0000001.txt").write_text(
        ANNOTATION_TEXT, encoding="utf-8"
    )
    return source_dir


def test_convert_visdrone_det_split_writes_manifest_labels_and_annotations(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path
    source_dir = make_source_split(tmp_path)
    output_root = repo_root / "data" / "raw" / "visdrone_det"

    manifest = convert_visdrone_det_split(
        source_dir,
        "val",
        output_root,
        repo_root,
        "https://example.invalid/VisDrone2019-DET-val.zip",
    )

    assert manifest.dataset == "visdrone_det"
    assert manifest.split == "val"
    assert len(manifest.samples) == 1
    sample = manifest.samples[0]
    assert sample.original_id == "0000001_00000_d_0000001"
    assert sample.width_px == 400
    assert sample.height_px == 300
    assert sample.box_count == 1
    assert sample.ignored_region_count == 1
    assert sample.sha256.startswith("sha256:")

    assert sample.relative_label_path is not None
    label_path = repo_root / sample.relative_label_path
    lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    class_id, cx, cy, w, h = lines[0].split()
    assert class_id == "0"  # category 1 (pedestrian) -> YOLO class 0
    assert pytest.approx(float(cx), abs=1e-4) == 110 / 400
    assert pytest.approx(float(cy), abs=1e-4) == 120 / 300
    assert pytest.approx(float(w), abs=1e-4) == 20 / 400
    assert pytest.approx(float(h), abs=1e-4) == 40 / 300

    assert sample.relative_annotation_path is not None
    annotation_payload = json.loads(
        (repo_root / sample.relative_annotation_path).read_text(encoding="utf-8")
    )
    assert len(annotation_payload["boxes"]) == 3

    assert sample.relative_image_path is not None
    assert (repo_root / sample.relative_image_path).exists()
    assert not (source_dir / "images" / "0000001_00000_d_0000001.jpg").exists()


def test_convert_visdrone_det_split_rejects_empty_source(tmp_path: Path) -> None:
    source_dir = tmp_path / "source" / "VisDrone2019-DET-val"
    (source_dir / "images").mkdir(parents=True)
    (source_dir / "annotations").mkdir(parents=True)

    with pytest.raises(ValueError, match="no images found"):
        convert_visdrone_det_split(
            source_dir, "val", tmp_path / "out", tmp_path, "https://example.invalid"
        )
