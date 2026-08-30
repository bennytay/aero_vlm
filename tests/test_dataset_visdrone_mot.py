from pathlib import Path

import pytest
from PIL import Image

from wam_drones.dataset.visdrone_mot import (
    convert_visdrone_mot_split,
    parse_visdrone_mot_annotation,
)

ANNOTATION_TEXT = """\
1,1,0,0,50,50,1,0,0,0
1,2,100,100,20,40,1,1,1,2
2,2,105,102,20,40,1,1,1,2
"""


def test_parse_visdrone_mot_annotation_reads_all_fields() -> None:
    boxes = parse_visdrone_mot_annotation(ANNOTATION_TEXT)

    assert len(boxes) == 3
    ignored, frame1_ped, frame2_ped = boxes
    assert ignored.frame_index == 1
    assert ignored.is_ignored_region
    assert frame1_ped.frame_index == 1
    assert frame1_ped.target_id == 2
    assert frame1_ped.is_trainable
    assert frame2_ped.frame_index == 2
    assert frame2_ped.bbox_xyxy_px == (105.0, 102.0, 125.0, 142.0)


def test_parse_visdrone_mot_annotation_rejects_malformed_row() -> None:
    with pytest.raises(ValueError, match="expected 10 fields"):
        parse_visdrone_mot_annotation("1,1,0,0,50,50,1,0,0\n")


def make_source_split(external_dir: Path) -> Path:
    # Deliberately outside repo_root, like a real manually-downloaded
    # archive extracted to ~/Downloads: catches converters that forget to
    # copy frames into the project tree before computing a repo-relative
    # path (see the visdrone_mot.py/uavdt.py fix for the real bug this
    # caught).
    source_dir = external_dir / "VisDrone2019-MOT-train"
    sequence_dir = source_dir / "sequences" / "uav0000001_00000_v"
    sequence_dir.mkdir(parents=True)
    (source_dir / "annotations").mkdir(parents=True)
    Image.new("RGB", (200, 100), color="red").save(sequence_dir / "0000001.jpg")
    Image.new("RGB", (200, 100), color="red").save(sequence_dir / "0000002.jpg")
    (source_dir / "annotations" / "uav0000001_00000_v.txt").write_text(
        ANNOTATION_TEXT, encoding="utf-8"
    )
    return source_dir


def test_convert_visdrone_mot_split_builds_per_frame_manifest(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    repo_root = tmp_path
    source_dir = make_source_split(tmp_path_factory.mktemp("external"))

    manifest = convert_visdrone_mot_split(
        source_dir,
        "train",
        repo_root / "data" / "raw" / "visdrone_mot",
        repo_root,
        "manual-download://visdrone-mot-train",
    )

    assert manifest.dataset == "visdrone_mot"
    assert len(manifest.samples) == 2
    by_frame = {sample.frame_number: sample for sample in manifest.samples}
    assert by_frame[1].sequence_id == "uav0000001_00000_v"
    assert by_frame[1].box_count == 1  # only the trainable pedestrian box
    assert by_frame[1].ignored_region_count == 1
    assert by_frame[2].box_count == 1
    assert by_frame[2].ignored_region_count == 0
    for sample in manifest.samples:
        assert sample.relative_label_path is not None
        assert (repo_root / sample.relative_label_path).exists()


def test_convert_visdrone_mot_split_rejects_empty_sequences(tmp_path: Path) -> None:
    source_dir = tmp_path / "source" / "VisDrone2019-MOT-train"
    (source_dir / "sequences").mkdir(parents=True)
    (source_dir / "annotations").mkdir(parents=True)

    with pytest.raises(ValueError, match="no sequences found"):
        convert_visdrone_mot_split(
            source_dir, "train", tmp_path / "out", tmp_path, "https://example.invalid"
        )
