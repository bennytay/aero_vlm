"""UAVDT ground-truth conversion from a locally extracted directory.

UAVDT has no scriptable public mirror (distributed via Google Drive/OneDrive
links from the dataset paper page), so this operates on an
already-downloaded-and-extracted directory, like `visdrone_mot.py`.

The parser follows the UAVDT-benchmark toolkit's documented ground-truth
format: `<seq>_gt_whole.txt` rows are
`frame_index,target_id,bbox_left,bbox_top,bbox_width,bbox_height,
out_of_view,occlusion,object_category` (object_category: 1=car, 2=truck,
3=bus); `<seq>_gt_ignore.txt` rows are
`frame_index,bbox_left,bbox_top,bbox_width,bbox_height` distractor/ignore
regions. This has not yet been validated against real UAVDT files (none were
available while building it) — parsing is strict, so a column-count mismatch
raises instead of silently mis-parsing; re-check both formats against a real
downloaded sequence before trusting the output for training.
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from wam_drones.dataset.manifest import DatasetManifest, DatasetSplit, ImageRecord
from wam_drones.detection.vocabulary import DETECTION_ID_BY_LABEL, DetectionLabel
from wam_drones.net import file_sha256

ANNOTATION_CONVERSION_VERSION = "uavdt_v1"
GT_WHOLE_FIELD_COUNT = 9
GT_IGNORE_FIELD_COUNT = 5

UAVDT_CATEGORY_TO_LABEL: dict[int, DetectionLabel] = {
    1: DetectionLabel.CAR,
    2: DetectionLabel.TRUCK,
    3: DetectionLabel.BUS,
}


@dataclass(frozen=True)
class UavdtBox:
    """One native UAVDT `_gt_whole.txt` annotation row for a single frame."""

    frame_index: int
    target_id: int
    bbox_xyxy_px: tuple[float, float, float, float]
    out_of_view: int
    occlusion: int
    category: int

    @property
    def label(self) -> DetectionLabel | None:
        """The vocabulary_detection_v1 label, or None for an unknown category."""
        return UAVDT_CATEGORY_TO_LABEL.get(self.category)


@dataclass(frozen=True)
class UavdtIgnoreRegion:
    """One native UAVDT `_gt_ignore.txt` distractor/ignore-region row."""

    frame_index: int
    bbox_xyxy_px: tuple[float, float, float, float]


def parse_uavdt_gt_whole(text: str) -> tuple[UavdtBox, ...]:
    """Parse one UAVDT `<seq>_gt_whole.txt` file's contents."""
    boxes: list[UavdtBox] = []
    for line_number, raw_line in enumerate(text.strip().splitlines(), start=1):
        line = raw_line.strip().rstrip(",")
        if not line:
            continue
        fields = line.split(",")
        if len(fields) != GT_WHOLE_FIELD_COUNT:
            raise ValueError(
                f"line {line_number}: expected {GT_WHOLE_FIELD_COUNT} fields, "
                f"got {len(fields)}: {raw_line!r}"
            )
        (
            frame_index,
            target_id,
            left,
            top,
            width,
            height,
            out_of_view,
            occlusion,
            category,
        ) = (int(field) for field in fields)
        if width < 0 or height < 0:
            raise ValueError(f"line {line_number}: negative box size: {raw_line!r}")
        boxes.append(
            UavdtBox(
                frame_index=frame_index,
                target_id=target_id,
                bbox_xyxy_px=(
                    float(left),
                    float(top),
                    float(left + width),
                    float(top + height),
                ),
                out_of_view=out_of_view,
                occlusion=occlusion,
                category=category,
            )
        )
    return tuple(boxes)


def parse_uavdt_gt_ignore(text: str) -> tuple[UavdtIgnoreRegion, ...]:
    """Parse one UAVDT `<seq>_gt_ignore.txt` file's contents."""
    regions: list[UavdtIgnoreRegion] = []
    for line_number, raw_line in enumerate(text.strip().splitlines(), start=1):
        line = raw_line.strip().rstrip(",")
        if not line:
            continue
        fields = line.split(",")
        if len(fields) != GT_IGNORE_FIELD_COUNT:
            raise ValueError(
                f"line {line_number}: expected {GT_IGNORE_FIELD_COUNT} fields, "
                f"got {len(fields)}: {raw_line!r}"
            )
        frame_index, left, top, width, height = (int(field) for field in fields)
        if width < 0 or height < 0:
            raise ValueError(f"line {line_number}: negative box size: {raw_line!r}")
        regions.append(
            UavdtIgnoreRegion(
                frame_index=frame_index,
                bbox_xyxy_px=(
                    float(left),
                    float(top),
                    float(left + width),
                    float(top + height),
                ),
            )
        )
    return tuple(regions)


def _yolo_line(box: UavdtBox, width_px: int, height_px: int) -> str:
    label = box.label
    if label is None:
        raise ValueError(f"unknown UAVDT category: {box.category}")
    x_min, y_min, x_max, y_max = box.bbox_xyxy_px
    dw, dh = 1.0 / width_px, 1.0 / height_px
    x_center = (x_min + x_max) / 2 * dw
    y_center = (y_min + y_max) / 2 * dh
    box_width = (x_max - x_min) * dw
    box_height = (y_max - y_min) * dh
    class_id = DETECTION_ID_BY_LABEL[label] - 1
    return f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"


def convert_uavdt_sequence(
    sequence_dir: Path,
    gt_whole_path: Path,
    gt_ignore_path: Path,
    sequence_id: str,
    split: DatasetSplit,
    output_root: Path,
    repo_root: Path,
    source_url: str,
) -> list[ImageRecord]:
    """Convert every frame of one UAVDT sequence into manifest rows + labels."""
    frame_paths = sorted(sequence_dir.glob("*.jpg"))
    if not frame_paths:
        raise ValueError(f"no frames found under {sequence_dir}")

    boxes_by_frame: dict[int, list[UavdtBox]] = defaultdict(list)
    if gt_whole_path.exists():
        for box in parse_uavdt_gt_whole(gt_whole_path.read_text(encoding="utf-8")):
            boxes_by_frame[box.frame_index].append(box)

    ignored_by_frame: dict[int, list[UavdtIgnoreRegion]] = defaultdict(list)
    if gt_ignore_path.exists():
        for region in parse_uavdt_gt_ignore(gt_ignore_path.read_text(encoding="utf-8")):
            ignored_by_frame[region.frame_index].append(region)

    split_output = output_root / split / sequence_id
    images_dir = split_output / "images"
    labels_dir = split_output / "labels"
    annotations_full_dir = split_output / "annotations_full"
    for directory in (images_dir, labels_dir, annotations_full_dir):
        directory.mkdir(parents=True, exist_ok=True)

    samples: list[ImageRecord] = []
    for frame_path in frame_paths:
        frame_index = int(frame_path.stem.removeprefix("img"))
        boxes = boxes_by_frame.get(frame_index, [])
        ignored_regions = ignored_by_frame.get(frame_index, [])

        destination_image = images_dir / frame_path.name
        if not destination_image.exists():
            shutil.copy2(frame_path, destination_image)

        with Image.open(destination_image) as image:
            width_px, height_px = image.size

        yolo_lines = [_yolo_line(box, width_px, height_px) for box in boxes]
        label_path = labels_dir / f"{frame_path.stem}.txt"
        label_path.write_text(
            "\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8"
        )

        annotation_path_full = annotations_full_dir / f"{frame_path.stem}.json"
        annotation_path_full.write_text(
            json.dumps(
                {
                    "boxes": [
                        {
                            "target_id": box.target_id,
                            "bbox_xyxy_px": list(box.bbox_xyxy_px),
                            "category": box.category,
                            "label": box.label.value if box.label else None,
                            "out_of_view": box.out_of_view,
                            "occlusion": box.occlusion,
                        }
                        for box in boxes
                    ],
                    "ignored_regions": [
                        {"bbox_xyxy_px": list(region.bbox_xyxy_px)}
                        for region in ignored_regions
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        samples.append(
            ImageRecord(
                dataset="uavdt",
                split=split,
                source_url=source_url,
                original_id=f"{sequence_id}_{frame_path.stem}",
                relative_image_path=destination_image.relative_to(
                    repo_root
                ).as_posix(),
                sha256=f"sha256:{file_sha256(destination_image)}",
                width_px=width_px,
                height_px=height_px,
                sequence_id=sequence_id,
                frame_number=frame_index,
                annotation_conversion_version=ANNOTATION_CONVERSION_VERSION,
                relative_label_path=label_path.relative_to(repo_root).as_posix(),
                relative_annotation_path=annotation_path_full.relative_to(
                    repo_root
                ).as_posix(),
                box_count=len(boxes),
                ignored_region_count=len(ignored_regions),
            )
        )

    return samples


def convert_uavdt_split(
    source_dir: Path,
    split: DatasetSplit,
    output_root: Path,
    repo_root: Path,
    source_url: str,
) -> DatasetManifest:
    """Convert every sequence in one extracted UAVDT split.

    Expects `source_dir/UAV-benchmark-M/<sequence>/*.jpg` frames and
    `source_dir/GT/<sequence>_gt_whole.txt` / `_gt_ignore.txt` labels.
    """
    sequences_dir = source_dir / "UAV-benchmark-M"
    gt_dir = source_dir / "GT"
    sequence_dirs = sorted(
        path for path in sequences_dir.iterdir() if path.is_dir()
    )
    if not sequence_dirs:
        raise ValueError(f"no sequences found under {sequences_dir}")

    samples: list[ImageRecord] = []
    for sequence_dir in sequence_dirs:
        sequence_id = sequence_dir.name
        samples.extend(
            convert_uavdt_sequence(
                sequence_dir,
                gt_dir / f"{sequence_id}_gt_whole.txt",
                gt_dir / f"{sequence_id}_gt_ignore.txt",
                sequence_id,
                split,
                output_root,
                repo_root,
                source_url,
            )
        )

    return DatasetManifest(
        dataset="uavdt",
        split=split,
        created_utc=datetime.now(UTC),
        samples=tuple(samples),
    )
