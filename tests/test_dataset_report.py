from datetime import UTC, datetime
from pathlib import Path

from wam_drones.dataset.manifest import DatasetManifest, ImageRecord
from wam_drones.dataset.report import build_dataset_report


def write_label(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")


def make_sample(
    original_id: str, label_path: str, width_px: int, height_px: int, box_count: int
) -> ImageRecord:
    return ImageRecord(
        dataset="visdrone_det",
        split="val",
        source_url="https://example.invalid",
        original_id=original_id,
        relative_image_path=f"images/{original_id}.jpg",
        sha256="sha256:" + "d" * 64,
        width_px=width_px,
        height_px=height_px,
        annotation_conversion_version="v1",
        relative_label_path=label_path,
        box_count=box_count,
        ignored_region_count=0,
    )


def test_build_dataset_report_counts_classes_and_small_box_shrink(
    tmp_path: Path,
) -> None:
    # A 640x640 image with one 32x32 "car" (class id 3) box: it stays >=8px
    # at both 640 (scale 1.0 -> 32px) and 320 (scale 0.5 -> 16px). A tiny
    # 4x4 "car" box stays 4x4 at 640 (not < 4px, but < 8px) and shrinks to
    # 2x2 at 320 (< 4px, and so also < 8px).
    write_label(
        tmp_path / "labels" / "a.txt",
        [
            f"3 0.5 0.5 {32 / 640:.6f} {32 / 640:.6f}",
            f"3 0.2 0.2 {4 / 640:.6f} {4 / 640:.6f}",
        ],
    )
    manifest = DatasetManifest(
        dataset="visdrone_det",
        split="val",
        created_utc=datetime(2026, 8, 31, tzinfo=UTC),
        samples=(make_sample("a", "labels/a.txt", 640, 640, 2),),
    )

    report = build_dataset_report(manifest, tmp_path, input_sizes=(640, 320))

    assert report.box_count == 2
    assert report.class_counts == {"car": 2}
    assert report.shrink_by_input_size[640].below_4px == 0
    assert report.shrink_by_input_size[640].below_8px == 1
    assert report.shrink_by_input_size[320].below_4px == 1
    assert report.shrink_by_input_size[320].below_8px == 1


def test_build_dataset_report_handles_images_with_no_boxes(tmp_path: Path) -> None:
    write_label(tmp_path / "labels" / "empty.txt", [])
    manifest = DatasetManifest(
        dataset="visdrone_det",
        split="val",
        created_utc=datetime(2026, 8, 31, tzinfo=UTC),
        samples=(make_sample("empty", "labels/empty.txt", 640, 480, 0),),
    )

    report = build_dataset_report(manifest, tmp_path)

    assert report.box_count == 0
    assert report.class_counts == {}
    assert all(
        counts.below_4px == 0 and counts.below_8px == 0
        for counts in report.shrink_by_input_size.values()
    )


def test_report_write_produces_json_and_markdown(tmp_path: Path) -> None:
    write_label(
        tmp_path / "labels" / "a.txt", [f"3 0.5 0.5 {32 / 640:.6f} {32 / 640:.6f}"]
    )
    manifest = DatasetManifest(
        dataset="visdrone_det",
        split="val",
        created_utc=datetime(2026, 8, 31, tzinfo=UTC),
        samples=(make_sample("a", "labels/a.txt", 640, 640, 1),),
    )
    report = build_dataset_report(manifest, tmp_path, input_sizes=(640,))

    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    report.write(json_path, markdown_path)

    assert json_path.exists()
    assert "car" in markdown_path.read_text(encoding="utf-8")
