"""VisDrone-MOT/VID conversion from a locally extracted sequence directory.

VisDrone-MOT and VisDrone-VID have no scriptable public mirror (unlike
VisDrone-DET); the official sources are Google Drive/OneDrive/BaiduDisk
links in https://github.com/VisDrone/VisDrone-Dataset. This module operates
on an already-downloaded-and-extracted directory, matching the manual-import
pattern `scripts/prepare_phase1_fixtures.py` already uses.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from PIL import Image

from wam_drones.dataset.manifest import DatasetManifest, DatasetSplit, ImageRecord
from wam_drones.net import file_sha256

ANNOTATION_CONVERSION_VERSION = "visdrone_mot_v1"
IGNORED_REGION_CATEGORY = 0
TRAINABLE_CATEGORIES = frozenset(range(1, 11))
ANNOTATION_FIELD_COUNT = 10

SequenceDataset = Literal["visdrone_mot", "visdrone_vid"]


@dataclass(frozen=True)
class VisDroneMotBox:
    """One native VisDrone-MOT/VID annotation row for a single frame."""

    frame_index: int
    target_id: int
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


def parse_visdrone_mot_annotation(text: str) -> tuple[VisDroneMotBox, ...]:
    """Parse one VisDrone-MOT/VID sequence-level `.txt` annotation file."""
    boxes: list[VisDroneMotBox] = []
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
        (
            frame_index,
            target_id,
            left,
            top,
            width,
            height,
            score,
            category,
            truncation,
            occlusion,
        ) = (int(field) for field in fields)
        if width < 0 or height < 0:
            raise ValueError(f"line {line_number}: negative box size: {raw_line!r}")
        boxes.append(
            VisDroneMotBox(
                frame_index=frame_index,
                target_id=target_id,
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


def _yolo_line(box: VisDroneMotBox, width_px: int, height_px: int) -> str:
    x_min, y_min, x_max, y_max = box.bbox_xyxy_px
    dw, dh = 1.0 / width_px, 1.0 / height_px
    x_center = (x_min + x_max) / 2 * dw
    y_center = (y_min + y_max) / 2 * dh
    box_width = (x_max - x_min) * dw
    box_height = (y_max - y_min) * dh
    class_id = box.category - 1
    return f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"


def convert_visdrone_mot_sequence(
    sequence_dir: Path,
    annotation_path: Path,
    sequence_id: str,
    split: DatasetSplit,
    output_root: Path,
    repo_root: Path,
    source_url: str,
    dataset: SequenceDataset = "visdrone_mot",
) -> list[ImageRecord]:
    """Convert every frame of one sequence into manifest rows + YOLO labels."""
    frame_paths = sorted(sequence_dir.glob("*.jpg"))
    if not frame_paths:
        raise ValueError(f"no frames found under {sequence_dir}")

    boxes_by_frame: dict[int, list[VisDroneMotBox]] = defaultdict(list)
    if annotation_path.exists():
        for box in parse_visdrone_mot_annotation(
            annotation_path.read_text(encoding="utf-8")
        ):
            boxes_by_frame[box.frame_index].append(box)

    split_output = output_root / split / sequence_id
    labels_dir = split_output / "labels"
    annotations_full_dir = split_output / "annotations_full"
    labels_dir.mkdir(parents=True, exist_ok=True)
    annotations_full_dir.mkdir(parents=True, exist_ok=True)

    samples: list[ImageRecord] = []
    for frame_path in frame_paths:
        frame_index = int(frame_path.stem)
        boxes = boxes_by_frame.get(frame_index, [])

        with Image.open(frame_path) as image:
            width_px, height_px = image.size

        trainable = [box for box in boxes if box.is_trainable]
        ignored_regions = [box for box in boxes if box.is_ignored_region]

        yolo_lines = [_yolo_line(box, width_px, height_px) for box in trainable]
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
                dataset=dataset,
                split=split,
                source_url=source_url,
                original_id=f"{sequence_id}_{frame_path.stem}",
                relative_image_path=frame_path.relative_to(repo_root).as_posix(),
                sha256=f"sha256:{file_sha256(frame_path)}",
                width_px=width_px,
                height_px=height_px,
                sequence_id=sequence_id,
                frame_number=frame_index,
                annotation_conversion_version=ANNOTATION_CONVERSION_VERSION,
                relative_label_path=label_path.relative_to(repo_root).as_posix(),
                relative_annotation_path=annotation_path_full.relative_to(
                    repo_root
                ).as_posix(),
                box_count=len(trainable),
                ignored_region_count=len(ignored_regions),
            )
        )

    return samples


def convert_visdrone_mot_split(
    source_dir: Path,
    split: DatasetSplit,
    output_root: Path,
    repo_root: Path,
    source_url: str,
    dataset: SequenceDataset = "visdrone_mot",
) -> DatasetManifest:
    """Convert every sequence in one extracted VisDrone-MOT/VID split."""
    sequences_dir = source_dir / "sequences"
    annotations_dir = source_dir / "annotations"
    sequence_dirs = sorted(
        path for path in sequences_dir.iterdir() if path.is_dir()
    )
    if not sequence_dirs:
        raise ValueError(f"no sequences found under {sequences_dir}")

    samples: list[ImageRecord] = []
    for sequence_dir in sequence_dirs:
        sequence_id = sequence_dir.name
        samples.extend(
            convert_visdrone_mot_sequence(
                sequence_dir,
                annotations_dir / f"{sequence_id}.txt",
                sequence_id,
                split,
                output_root,
                repo_root,
                source_url,
                dataset,
            )
        )

    return DatasetManifest(
        dataset=dataset,
        split=split,
        created_utc=datetime.now(UTC),
        samples=tuple(samples),
    )
