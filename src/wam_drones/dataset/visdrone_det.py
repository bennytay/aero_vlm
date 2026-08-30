"""VisDrone2019-DET download and conversion to manifest + YOLO format."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

from wam_drones.dataset.archive_config import DatasetArchiveConfig
from wam_drones.dataset.manifest import DatasetManifest, DatasetSplit, ImageRecord
from wam_drones.net import download_with_sha256, file_sha256

ANNOTATION_CONVERSION_VERSION = "visdrone_det_v1"
IGNORED_REGION_CATEGORY = 0
OTHERS_CATEGORY = 11
TRAINABLE_CATEGORIES = frozenset(range(1, 11))
ANNOTATION_FIELD_COUNT = 8

SPLIT_TO_ARCHIVE_NAME = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
    "test": "VisDrone2019-DET-test-dev",
}


@dataclass(frozen=True)
class VisDroneBox:
    """One native VisDrone-DET annotation row, before class filtering."""

    bbox_xyxy_px: tuple[float, float, float, float]
    score: int
    category: int
    truncation: int
    occlusion: int

    @property
    def is_ignored_region(self) -> bool:
        """A drawn ignore-region mask, distinct from a per-object score flag."""
        return self.category == IGNORED_REGION_CATEGORY

    @property
    def is_trainable(self) -> bool:
        """Whether this row maps onto a vocabulary_detection_v1 class."""
        return self.category in TRAINABLE_CATEGORIES and self.score != 0


def parse_visdrone_det_annotation(text: str) -> tuple[VisDroneBox, ...]:
    """Parse one VisDrone-DET `.txt` annotation file's contents."""
    boxes: list[VisDroneBox] = []
    for line_number, raw_line in enumerate(text.strip().splitlines(), start=1):
        line = raw_line.strip().rstrip(",")
        if not line:
            continue
        fields = line.split(",")
        if len(fields) != ANNOTATION_FIELD_COUNT:
            raise ValueError(
                f"line {line_number}: expected {ANNOTATION_FIELD_COUNT} fields, "
                f"got {len(fields)}: {raw_line!r}"
            )
        left, top, width, height, score, category, truncation, occlusion = (
            int(field) for field in fields
        )
        if width < 0 or height < 0:
            raise ValueError(f"line {line_number}: negative box size: {raw_line!r}")
        boxes.append(
            VisDroneBox(
                bbox_xyxy_px=(
                    float(left),
                    float(top),
                    float(left + width),
                    float(top + height),
                ),
                score=score,
                category=category,
                truncation=truncation,
                occlusion=occlusion,
            )
        )
    return tuple(boxes)


def download_visdrone_det_split(
    split: DatasetSplit, dest_dir: Path, config: DatasetArchiveConfig
) -> tuple[Path, str]:
    """Download, verify, and extract one split.

    Returns the extracted directory and the archive's actual SHA-256, so a
    first-time caller can pin it into the checked-in config.
    """
    archive_name = SPLIT_TO_ARCHIVE_NAME[split]
    split_config = config.splits[split]
    archive_path = dest_dir / "_archives" / f"{archive_name}.zip"
    sha256_hex = download_with_sha256(
        split_config.url, archive_path, split_config.sha256
    )
    extract_dir = dest_dir / "_extracted" / archive_name
    if not extract_dir.exists():
        with ZipFile(archive_path) as archive:
            archive.extractall(dest_dir / "_extracted")
    return extract_dir, sha256_hex


def _yolo_line(box: VisDroneBox, width_px: int, height_px: int) -> str:
    x_min, y_min, x_max, y_max = box.bbox_xyxy_px
    dw, dh = 1.0 / width_px, 1.0 / height_px
    x_center = (x_min + x_max) / 2 * dw
    y_center = (y_min + y_max) / 2 * dh
    box_width = (x_max - x_min) * dw
    box_height = (y_max - y_min) * dh
    class_id = box.category - 1
    return f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"


def convert_visdrone_det_split(
    source_dir: Path,
    split: DatasetSplit,
    output_root: Path,
    repo_root: Path,
    source_url: str,
) -> DatasetManifest:
    """Convert one extracted VisDrone-DET split into manifest + YOLO labels."""
    images_source_dir = source_dir / "images"
    annotations_dir = source_dir / "annotations"
    image_paths = sorted(images_source_dir.glob("*.jpg"))
    if not image_paths:
        raise ValueError(f"no images found under {images_source_dir}")

    split_output = output_root / split
    images_dir = split_output / "images"
    labels_dir = split_output / "labels"
    annotations_full_dir = split_output / "annotations_full"
    for directory in (images_dir, labels_dir, annotations_full_dir):
        directory.mkdir(parents=True, exist_ok=True)

    samples: list[ImageRecord] = []
    for image_path in image_paths:
        original_id = image_path.stem
        annotation_path = annotations_dir / f"{original_id}.txt"
        boxes = (
            parse_visdrone_det_annotation(
                annotation_path.read_text(encoding="utf-8")
            )
            if annotation_path.exists()
            else ()
        )

        with Image.open(image_path) as image:
            width_px, height_px = image.size

        destination_image = images_dir / image_path.name
        if not destination_image.exists():
            shutil.move(str(image_path), str(destination_image))

        trainable = [box for box in boxes if box.is_trainable]
        ignored_regions = [box for box in boxes if box.is_ignored_region]

        yolo_lines = [_yolo_line(box, width_px, height_px) for box in trainable]
        label_path = labels_dir / f"{original_id}.txt"
        label_path.write_text(
            "\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8"
        )

        annotation_path_full = annotations_full_dir / f"{original_id}.json"
        annotation_path_full.write_text(
            json.dumps(
                {
                    "boxes": [
                        {
                            "bbox_xyxy_px": list(box.bbox_xyxy_px),
                            "score": box.score,
                            "category": box.category,
                            "truncation": box.truncation,
                            "occlusion": box.occlusion,
                            "is_ignored_region": box.is_ignored_region,
                            "is_trainable": box.is_trainable,
                        }
                        for box in boxes
                    ]
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        samples.append(
            ImageRecord(
                dataset="visdrone_det",
                split=split,
                source_url=source_url,
                original_id=original_id,
                relative_image_path=destination_image.relative_to(
                    repo_root
                ).as_posix(),
                sha256=f"sha256:{file_sha256(destination_image)}",
                width_px=width_px,
                height_px=height_px,
                annotation_conversion_version=ANNOTATION_CONVERSION_VERSION,
                relative_label_path=label_path.relative_to(repo_root).as_posix(),
                relative_annotation_path=annotation_path_full.relative_to(
                    repo_root
                ).as_posix(),
                box_count=len(trainable),
                ignored_region_count=len(ignored_regions),
            )
        )

    return DatasetManifest(
        dataset="visdrone_det",
        split=split,
        created_utc=datetime.now(UTC),
        samples=tuple(samples),
    )
