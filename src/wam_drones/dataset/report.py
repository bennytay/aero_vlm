"""Class counts, box-area histogram, and post-resize shrink report.

Reads the YOLO label files a converter already wrote (0-indexed class IDs
that equal `vocabulary_detection_v1` label IDs minus one for every dataset),
so this works uniformly across VisDrone-DET/MOT/VID and UAVDT without any
dataset-specific parsing.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from wam_drones.dataset.manifest import DatasetManifest
from wam_drones.detection.vocabulary import DETECTION_LABEL_BY_ID

DEFAULT_INPUT_SIZES: tuple[int, ...] = (640, 512, 416, 320)
SMALL_BOX_THRESHOLDS_PX: tuple[int, ...] = (4, 8)
AREA_HISTOGRAM_EDGES_PX: tuple[float, ...] = (
    0,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    float("inf"),
)


@dataclass(frozen=True)
class BoxSizeThresholdCounts:
    """How many boxes fall below each small-box pixel threshold."""

    below_4px: int
    below_8px: int


@dataclass
class DatasetReport:
    """Aggregated class, area, and post-resize shrink statistics."""

    dataset: str
    split: str
    image_count: int
    box_count: int
    ignored_region_count: int
    class_counts: dict[str, int]
    area_histogram_px2: dict[str, int]
    shrink_by_input_size: dict[int, BoxSizeThresholdCounts]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation."""
        return {
            "dataset": self.dataset,
            "split": self.split,
            "image_count": self.image_count,
            "box_count": self.box_count,
            "ignored_region_count": self.ignored_region_count,
            "class_counts": self.class_counts,
            "area_histogram_px2": self.area_histogram_px2,
            "shrink_by_input_size": {
                str(size): {
                    "below_4px": counts.below_4px,
                    "below_8px": counts.below_8px,
                }
                for size, counts in sorted(self.shrink_by_input_size.items())
            },
        }

    def to_markdown(self) -> str:
        """Return a short human-readable summary table."""
        lines = [
            f"# {self.dataset} / {self.split} resize report",
            "",
            f"- Images: {self.image_count}",
            f"- Boxes: {self.box_count}",
            f"- Ignored regions: {self.ignored_region_count}",
            "",
            "## Class counts",
            "",
            "| Class | Count |",
            "|---|---|",
        ]
        for label, count in sorted(self.class_counts.items()):
            lines.append(f"| {label} | {count} |")
        lines += [
            "",
            "## Boxes below size threshold after letterbox resize",
            "",
            "| Input size | < 4px min dim | < 8px min dim |",
            "|---|---|---|",
        ]
        for size, counts in sorted(self.shrink_by_input_size.items()):
            lines.append(f"| {size} | {counts.below_4px} | {counts.below_8px} |")
        lines.append("")
        return "\n".join(lines)

    def write(self, json_path: Path, markdown_path: Path) -> None:
        """Write both the machine-readable and human-readable reports."""
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(self.to_markdown(), encoding="utf-8")


def _area_bucket(area_px2: float) -> str:
    for low, high in pairwise(AREA_HISTOGRAM_EDGES_PX):
        if low <= area_px2 < high:
            low_label = int(low)
            high_label = "inf" if high == float("inf") else int(high)
            return f"{low_label}-{high_label}"
    raise ValueError(f"box area out of histogram range: {area_px2}")


def build_dataset_report(
    manifest: DatasetManifest,
    repo_root: Path,
    input_sizes: tuple[int, ...] = DEFAULT_INPUT_SIZES,
) -> DatasetReport:
    """Aggregate class, area, and letterbox-shrink statistics for a manifest."""
    class_counts: Counter[str] = Counter()
    area_histogram: Counter[str] = Counter()
    shrink_counts: dict[int, list[int]] = {size: [0, 0] for size in input_sizes}
    total_boxes = 0

    for sample in manifest.samples:
        if sample.relative_label_path is None:
            continue
        text = (repo_root / sample.relative_label_path).read_text(
            encoding="utf-8"
        ).strip()
        if not text:
            continue
        for line in text.splitlines():
            class_id_str, _cx, _cy, w_str, h_str = line.split()
            label = DETECTION_LABEL_BY_ID[int(class_id_str) + 1]
            width_px = float(w_str) * sample.width_px
            height_px = float(h_str) * sample.height_px

            total_boxes += 1
            class_counts[label.value] += 1
            area_histogram[_area_bucket(width_px * height_px)] += 1

            for size in input_sizes:
                scale = min(size / sample.width_px, size / sample.height_px)
                min_dim_px = min(width_px, height_px) * scale
                if min_dim_px < SMALL_BOX_THRESHOLDS_PX[1]:
                    shrink_counts[size][1] += 1
                if min_dim_px < SMALL_BOX_THRESHOLDS_PX[0]:
                    shrink_counts[size][0] += 1

    return DatasetReport(
        dataset=manifest.dataset,
        split=manifest.split,
        image_count=len(manifest.samples),
        box_count=total_boxes,
        ignored_region_count=sum(
            sample.ignored_region_count for sample in manifest.samples
        ),
        class_counts=dict(class_counts),
        area_histogram_px2=dict(area_histogram),
        shrink_by_input_size={
            size: BoxSizeThresholdCounts(below_4px=counts[0], below_8px=counts[1])
            for size, counts in shrink_counts.items()
        },
    )
