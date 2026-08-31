"""Reproducible Phase 3 VisDrone fine-tuning and evaluation utilities."""

from __future__ import annotations

import json
import os
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw

from wam_drones.dataset.manifest import DatasetManifest, ImageRecord
from wam_drones.detection.checkpoint import ensure_checkpoint
from wam_drones.detection.model_config import load_model_config
from wam_drones.detection.vocabulary import DETECTION_ID_BY_LABEL

CLASS_NAMES = tuple(DETECTION_ID_BY_LABEL)
IOU_THRESHOLDS = tuple(round(0.5 + index * 0.05, 2) for index in range(10))
PRIMARY_IOU_THRESHOLD = 0.5
SMALL_AREA_PX2 = 32 * 32


@dataclass(frozen=True)
class Box:
    """One native-class target or prediction in original-image coordinates."""

    class_id: int | None
    xyxy: tuple[float, float, float, float]
    confidence: float | None = None
    occlusion: int | None = None

    @property
    def area(self) -> float:
        return max(0.0, self.xyxy[2] - self.xyxy[0]) * max(
            0.0, self.xyxy[3] - self.xyxy[1]
        )


@dataclass(frozen=True)
class ImageEvaluationInput:
    sample: ImageRecord
    targets: tuple[Box, ...]
    ignored_regions: tuple[Box, ...]


@dataclass(frozen=True)
class ReviewCandidate:
    kind: str
    original_id: str
    image_path: Path
    class_name: str
    xyxy: tuple[float, float, float, float]
    confidence: float | None
    area_px2: float
    occlusion: int | None


def set_deterministic_seed(seed: int) -> None:
    """Set supported RNGs and Torch deterministic options before a run."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    try:
        numpy = import_module("numpy")
        numpy.random.seed(seed)
    except ImportError:
        pass
    try:
        torch = import_module("torch")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def _manifest_hash(paths: Iterable[Path]) -> str:
    digest = sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _leaked_validation_ids(
    train_manifest: DatasetManifest,
    val_manifest: DatasetManifest,
    dedup_report: Path,
) -> tuple[str, ...]:
    """Find DET-val samples in near-duplicate groups that include DET-train."""
    payload = json.loads(dedup_report.read_text(encoding="utf-8"))
    train_ids = {sample.original_id for sample in train_manifest.samples}
    val_ids = {sample.original_id for sample in val_manifest.samples}
    excluded: set[str] = set()
    for group in payload.get("groups", []):
        ids = set(group["original_ids"])
        if ids & train_ids and ids & val_ids:
            excluded.update(ids & val_ids)
    return tuple(sorted(excluded))


def prepare_phase3_dataset(
    *,
    repo_root: Path,
    output_dir: Path,
    train_manifest_path: Path,
    val_manifest_path: Path,
    dedup_report_path: Path,
) -> dict[str, Any]:
    """Write run-local lists that exclude the known DET train/val leak."""
    train_manifest = DatasetManifest.read(train_manifest_path)
    val_manifest = DatasetManifest.read(val_manifest_path)
    if (train_manifest.dataset, train_manifest.split) != ("visdrone_det", "train"):
        raise ValueError("train manifest must be VisDrone-DET train")
    if (val_manifest.dataset, val_manifest.split) != ("visdrone_det", "val"):
        raise ValueError("validation manifest must be VisDrone-DET val")

    excluded = _leaked_validation_ids(train_manifest, val_manifest, dedup_report_path)
    if not excluded:
        raise ValueError("no DET train/val near-duplicate exclusions found")
    included_val = tuple(
        sample for sample in val_manifest.samples if sample.original_id not in excluded
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    train_list = output_dir / "train_images.txt"
    val_list = output_dir / "sequence_safe_val_images.txt"
    excluded_list = output_dir / "excluded_val_leak_images.txt"
    safe_manifest_path = output_dir / "sequence_safe_val_manifest.json"
    data_yaml = output_dir / "visdrone_det_phase3.yaml"

    def write_list(path: Path, samples: Sequence[ImageRecord]) -> None:
        path.write_text(
            "\n".join(
                str((repo_root / sample.relative_image_path).resolve())
                for sample in samples
            )
            + "\n",
            encoding="utf-8",
        )

    write_list(train_list, train_manifest.samples)
    write_list(val_list, included_val)
    write_list(
        excluded_list,
        tuple(
            sample
            for sample in val_manifest.samples
            if sample.original_id in excluded
        ),
    )
    DatasetManifest(
        dataset=val_manifest.dataset,
        split=val_manifest.split,
        created_utc=val_manifest.created_utc,
        samples=included_val,
    ).write(safe_manifest_path)
    data_yaml.write_text(
        "# Generated by wam-detect phase3-prepare; raw data is immutable.\n"
        f"train: {train_list}\nval: {val_list}\nnames:\n"
        + "".join(f"  {index}: {name}\n" for index, name in enumerate(CLASS_NAMES)),
        encoding="utf-8",
    )
    metadata = {
        "protocol": (
            "official DET train; official DET val excluding known train/val "
            "near-duplicate(s) for primary sequence-safe validation"
        ),
        "dedup_report": str(dedup_report_path.resolve()),
        "excluded_validation_original_ids": list(excluded),
        "train_image_count": len(train_manifest.samples),
        "sequence_safe_val_image_count": len(included_val),
        "source_manifest_hash": _manifest_hash(
            (train_manifest_path, val_manifest_path)
        ),
        "letterbox": {"input_size_px": 640, "stretch": False, "crop": False},
    }
    (output_dir / "dataset_protocol.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def load_evaluation_inputs(
    manifest_path: Path, repo_root: Path
) -> tuple[ImageEvaluationInput, ...]:
    """Load the preserved native annotations for metric and review work."""
    manifest = DatasetManifest.read(manifest_path)
    images: list[ImageEvaluationInput] = []
    for sample in manifest.samples:
        if sample.relative_annotation_path is None:
            raise ValueError(f"missing full annotation path for {sample.original_id}")
        payload = json.loads(
            (repo_root / sample.relative_annotation_path).read_text(encoding="utf-8")
        )
        targets: list[Box] = []
        ignored: list[Box] = []
        for raw in payload["boxes"]:
            box = Box(
                class_id=(int(raw["category"]) - 1) if raw["is_trainable"] else None,
                xyxy=tuple(float(value) for value in raw["bbox_xyxy_px"]),  # type: ignore[arg-type]
                occlusion=int(raw["occlusion"]),
            )
            if raw["is_trainable"]:
                targets.append(box)
            elif raw["is_ignored_region"]:
                ignored.append(box)
        images.append(ImageEvaluationInput(sample, tuple(targets), tuple(ignored)))
    return tuple(images)


def _iou(left: Box, right: Box) -> float:
    x1 = max(left.xyxy[0], right.xyxy[0])
    y1 = max(left.xyxy[1], right.xyxy[1])
    x2 = min(left.xyxy[2], right.xyxy[2])
    y2 = min(left.xyxy[3], right.xyxy[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left.xyxy[2] - left.xyxy[0]) * max(
        0.0, left.xyxy[3] - left.xyxy[1]
    )
    right_area = max(0.0, right.xyxy[2] - right.xyxy[0]) * max(
        0.0, right.xyxy[3] - right.xyxy[1]
    )
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _letterbox_scale(sample: ImageRecord, input_size_px: int) -> float:
    return min(input_size_px / sample.width_px, input_size_px / sample.height_px)


def _size_bucket(area_px2: float) -> str:
    if area_px2 < SMALL_AREA_PX2:
        return "small_lt_32px"
    if area_px2 < 96 * 96:
        return "medium_32_to_96px"
    return "large_ge_96px"


def _density_bucket(count: int) -> str:
    if count <= 1:
        return "1"
    if count <= 5:
        return "2_to_5"
    if count <= 15:
        return "6_to_15"
    return "16_plus"


def _occlusion_bucket(occlusion: int | None) -> str:
    if occlusion is None:
        return "unknown"
    return {0: "none", 1: "partial", 2: "heavy"}.get(occlusion, "unknown")


def _scene_id(sample: ImageRecord) -> str:
    """Use the leading source token as the available VisDrone scene/video proxy."""
    return sample.original_id.split("_", 1)[0]


def _average_precision(
    points: Sequence[tuple[float, bool]], targets: int
) -> float | None:
    if targets == 0:
        return None
    ordered = sorted(points, key=lambda point: point[0], reverse=True)
    tp = 0
    curve: list[tuple[float, float]] = []
    for index, (_, match) in enumerate(ordered, start=1):
        tp += int(match)
        curve.append((tp / index, tp / targets))
    return sum(
        max((precision for precision, recall in curve if recall >= level), default=0.0)
        for level in (index / 100 for index in range(101))
    ) / 101


def _matches_for_class_at_thresholds(
    images: Sequence[ImageEvaluationInput],
    predictions: Mapping[str, Sequence[Box]],
    class_id: int,
    iou_thresholds: Sequence[float],
    target_filter: Callable[[ImageEvaluationInput, Box], bool] | None = None,
) -> dict[float, tuple[list[tuple[float, bool]], int, int, int, int]]:
    """Match once per image/class, sharing overlap work across AP thresholds."""
    points_by_threshold: dict[float, list[tuple[float, bool]]] = {
        threshold: [] for threshold in iou_thresholds
    }
    true_positives = {threshold: 0 for threshold in iou_thresholds}
    false_positives = {threshold: 0 for threshold in iou_thresholds}
    target_count = 0
    for image in images:
        accepted = [
            target
            for target in image.targets
            if target.class_id == class_id
            and (target_filter is None or target_filter(image, target))
        ]
        excluded = [
            target
            for target in image.targets
            if target.class_id == class_id and target not in accepted
        ]
        target_count += len(accepted)
        ignored = (*image.ignored_regions, *excluded)
        current = sorted(
            (
                box
                for box in predictions.get(image.sample.original_id, ())
                if box.class_id == class_id
            ),
            key=lambda box: box.confidence or 0.0,
            reverse=True,
        )
        target_overlaps = [
            [_iou(prediction, target) for target in accepted]
            for prediction in current
        ]
        ignored_overlaps = [
            [_iou(prediction, box) for box in ignored] for prediction in current
        ]
        for threshold in iou_thresholds:
            unmatched = set(range(len(accepted)))
            for prediction_index, prediction in enumerate(current):
                matching = [
                    (index, target_overlaps[prediction_index][index])
                    for index in unmatched
                    if target_overlaps[prediction_index][index] >= threshold
                ]
                if matching:
                    match, _overlap = max(matching, key=lambda item: item[1])
                    unmatched.remove(match)
                    points_by_threshold[threshold].append(
                        (prediction.confidence or 0.0, True)
                    )
                    if (prediction.confidence or 0.0) >= 0.25:
                        true_positives[threshold] += 1
                elif not any(
                    overlap >= threshold
                    for overlap in ignored_overlaps[prediction_index]
                ):
                    points_by_threshold[threshold].append(
                        (prediction.confidence or 0.0, False)
                    )
                    if (prediction.confidence or 0.0) >= 0.25:
                        false_positives[threshold] += 1
    return {
        threshold: (
            points_by_threshold[threshold],
            target_count,
            true_positives[threshold],
            false_positives[threshold],
            target_count - true_positives[threshold],
        )
        for threshold in iou_thresholds
    }


def _matches_for_class(
    images: Sequence[ImageEvaluationInput],
    predictions: Mapping[str, Sequence[Box]],
    class_id: int,
    iou_threshold: float,
    target_filter: Callable[[ImageEvaluationInput, Box], bool] | None = None,
) -> tuple[list[tuple[float, bool]], int, int, int, int]:
    return _matches_for_class_at_thresholds(
        images, predictions, class_id, (iou_threshold,), target_filter
    )[iou_threshold]


def _metric_row(
    images: Sequence[ImageEvaluationInput],
    predictions: Mapping[str, Sequence[Box]],
    class_id: int | None = None,
    target_filter: Callable[[ImageEvaluationInput, Box], bool] | None = None,
) -> dict[str, Any]:
    class_ids = (class_id,) if class_id is not None else tuple(range(len(CLASS_NAMES)))
    ap_values: list[float] = []
    ap50_values: list[float] = []
    target_count = true_positives = false_positives = false_negatives = 0
    for current_class in class_ids:
        aps: list[float] = []
        primary_counts: (
            tuple[list[tuple[float, bool]], int, int, int, int] | None
        ) = None
        matches_by_threshold = _matches_for_class_at_thresholds(
            images, predictions, current_class, IOU_THRESHOLDS, target_filter
        )
        for threshold in IOU_THRESHOLDS:
            points, count, _tp, _fp, _fn = matches_by_threshold[threshold]
            ap = _average_precision(points, count)
            if ap is not None:
                aps.append(ap)
                if threshold == PRIMARY_IOU_THRESHOLD:
                    ap50_values.append(ap)
            if threshold == PRIMARY_IOU_THRESHOLD:
                primary_counts = (points, count, _tp, _fp, _fn)
        if aps:
            ap_values.append(sum(aps) / len(aps))
        if primary_counts is None:
            raise AssertionError("primary IoU threshold missing")
        _points, count, tp, fp, fn = primary_counts
        target_count += count
        true_positives += tp
        false_positives += fp
        false_negatives += fn
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    return {
        "map50_95": sum(ap_values) / len(ap_values) if ap_values else None,
        "ap50": sum(ap50_values) / len(ap50_values) if ap50_values else None,
        "precision": precision,
        "recall": true_positives / target_count if target_count else 0.0,
        "false_positives_per_frame": false_positives / len(images) if images else 0.0,
        "ground_truth_count": target_count,
        "true_positive_count": true_positives,
        "false_positive_count": false_positives,
        "false_negative_count": false_negatives,
        "frame_count": len(images),
    }


def _diagnostic_row(
    images: Sequence[ImageEvaluationInput],
    predictions: Mapping[str, Sequence[Box]],
    target_filter: Callable[[ImageEvaluationInput, Box], bool],
) -> dict[str, Any]:
    """Fast fixed-IoU diagnostic row for required failure breakdowns.

    Per-class mAP and AP-small retain the full COCO IoU sweep.  Breakdowns are
    diagnostic operating-point scores at the documented primary IoU=0.50,
    avoiding a misleadingly expensive re-evaluation of every scene slice.
    """
    target_count = true_positives = false_positives = false_negatives = 0
    for class_id in range(len(CLASS_NAMES)):
        _points, count, tp, fp, fn = _matches_for_class(
            images, predictions, class_id, PRIMARY_IOU_THRESHOLD, target_filter
        )
        target_count += count
        true_positives += tp
        false_positives += fp
        false_negatives += fn
    return {
        "iou_threshold": PRIMARY_IOU_THRESHOLD,
        "precision": (
            true_positives / (true_positives + false_positives)
            if true_positives + false_positives
            else 0.0
        ),
        "recall": true_positives / target_count if target_count else 0.0,
        "false_positives_per_frame": false_positives / len(images) if images else 0.0,
        "ground_truth_count": target_count,
        "true_positive_count": true_positives,
        "false_positive_count": false_positives,
        "false_negative_count": false_negatives,
        "frame_count": len(images),
    }


def evaluate_predictions(
    images: Sequence[ImageEvaluationInput],
    predictions: Mapping[str, Sequence[Box]],
    *,
    input_size_px: int = 640,
) -> dict[str, Any]:
    """Calculate all Phase 3 score and failure-breakdown requirements."""
    per_class: dict[str, Any] = {}
    for class_id, name in enumerate(CLASS_NAMES):
        row = _metric_row(images, predictions, class_id)
        row["ap_small"] = _metric_row(
            images,
            predictions,
            class_id,
            lambda image, target: (
                target.area * _letterbox_scale(image.sample, input_size_px) ** 2
                < SMALL_AREA_PX2
            ),
        )["map50_95"]
        per_class[name] = row

    def grouped(key: Callable[[ImageEvaluationInput, Box], str]) -> dict[str, Any]:
        buckets = sorted(
            {key(image, target) for image in images for target in image.targets}
        )
        result: dict[str, Any] = {}
        for bucket in buckets:
            def in_bucket(
                image: ImageEvaluationInput,
                target: Box,
                group: str = bucket,
            ) -> bool:
                return key(image, target) == group

            result[bucket] = _diagnostic_row(images, predictions, in_bucket)
        return result

    return {
        "metric_protocol": {
            "iou_thresholds": list(IOU_THRESHOLDS),
            "precision_recall_iou": PRIMARY_IOU_THRESHOLD,
            "confidence_threshold": 0.25,
            "small_definition": "letterboxed box area < 32^2 pixels at 640",
            "ignored_region_policy": (
                "predictions overlapping ignored or excluded-slice boxes at the "
                "active IoU threshold are ignored"
            ),
            "scene_definition": "leading VisDrone DET original_id source token",
        },
        "overall": _metric_row(images, predictions),
        "per_class": per_class,
        "breakdowns": {
            "box_size": grouped(
                lambda image, target: _size_bucket(
                    target.area * _letterbox_scale(image.sample, input_size_px) ** 2
                )
            ),
            "density": grouped(
                lambda image, target: _density_bucket(len(image.targets))
            ),
            "occlusion": grouped(
                lambda image, target: _occlusion_bucket(target.occlusion)
            ),
            "scene": grouped(lambda image, target: _scene_id(image.sample)),
        },
    }


def _review_candidates(
    images: Sequence[ImageEvaluationInput], predictions: Mapping[str, Sequence[Box]]
) -> tuple[list[ReviewCandidate], list[ReviewCandidate]]:
    false_negatives: list[ReviewCandidate] = []
    false_positives: list[ReviewCandidate] = []
    for image in images:
        used_targets: set[int] = set()
        for prediction in predictions.get(image.sample.original_id, ()):
            if (prediction.confidence or 0.0) < 0.25 or prediction.class_id is None:
                continue
            matches = [
                index
                for index, target in enumerate(image.targets)
                if index not in used_targets
                and target.class_id == prediction.class_id
                and _iou(target, prediction) >= PRIMARY_IOU_THRESHOLD
            ]
            if matches:
                used_targets.add(
                    max(
                        matches,
                        key=lambda index: _iou(image.targets[index], prediction),
                    )
                )
            elif not any(
                _iou(prediction, ignored) >= PRIMARY_IOU_THRESHOLD
                for ignored in image.ignored_regions
            ):
                false_positives.append(
                    ReviewCandidate(
                        "fp",
                        image.sample.original_id,
                        Path(image.sample.relative_image_path),
                        CLASS_NAMES[prediction.class_id],
                        prediction.xyxy,
                        prediction.confidence,
                        prediction.area,
                        None,
                    )
                )
        for index, target in enumerate(image.targets):
            if index not in used_targets:
                assert target.class_id is not None
                false_negatives.append(
                    ReviewCandidate(
                        "fn",
                        image.sample.original_id,
                        Path(image.sample.relative_image_path),
                        CLASS_NAMES[target.class_id],
                        target.xyxy,
                        None,
                        target.area,
                        target.occlusion,
                    )
                )
    return (
        sorted(
            false_negatives,
            key=lambda candidate: (
                candidate.area_px2,
                candidate.original_id,
                candidate.class_name,
            ),
        ),
        sorted(
            false_positives,
            key=lambda candidate: (
                -(candidate.confidence or 0.0),
                candidate.original_id,
                candidate.class_name,
            ),
        ),
    )


def write_manual_review_pack(
    images: Sequence[ImageEvaluationInput],
    predictions: Mapping[str, Sequence[Box]],
    *,
    repo_root: Path,
    output_dir: Path,
    count: int = 25,
) -> dict[str, Any]:
    """Render a deterministic 25-FN/25-FP human-review pack."""
    fns, fps = _review_candidates(images, predictions)
    if len(fns) < count or len(fps) < count:
        raise ValueError(
            f"review requires {count} FN and {count} FP candidates; found "
            f"{len(fns)} FN and {len(fps)} FP"
        )
    selected = [*fns[:count], *fps[:count]]
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    table = [
        "# Phase 3 manual error review",
        "",
        "Review all 50 samples before changing architecture. Mark each row "
        "reviewed and record the error cause.",
        "",
        "| ID | Type | Class | Confidence | Area px² | Occlusion | Status / cause |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for index, candidate in enumerate(selected, start=1):
        with Image.open(repo_root / candidate.image_path).convert("RGB") as image:
            draw = ImageDraw.Draw(image)
            color = "red" if candidate.kind == "fn" else "orange"
            draw.rectangle(candidate.xyxy, outline=color, width=3)
            draw.text(
                (candidate.xyxy[0], max(0, candidate.xyxy[1] - 12)),
                f"{candidate.kind.upper()} {candidate.class_name}",
                fill=color,
            )
            image_name = f"{index:02d}_{candidate.kind}_{candidate.original_id}.jpg"
            image.save(output_dir / image_name, quality=95)
        row = asdict(candidate)
        row["image_path"] = str(candidate.image_path)
        row.update({"image": image_name, "status": "pending", "cause": ""})
        rows.append(row)
        confidence = candidate.confidence if candidate.confidence is not None else "—"
        occlusion = candidate.occlusion if candidate.occlusion is not None else "—"
        table.append(
            f"| {index:02d} ({image_name}) | {candidate.kind.upper()} | "
            f"{candidate.class_name} | {confidence} | {candidate.area_px2:.1f} | "
            f"{occlusion} | pending |"
        )
    (output_dir / "review_manifest.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "MANUAL_REVIEW.md").write_text(
        "\n".join(table) + "\n", encoding="utf-8"
    )
    return {
        "false_negative_candidates": len(fns),
        "false_positive_candidates": len(fps),
        "selected_each": count,
        "status": "pending_human_review",
    }


def infer_predictions(
    checkpoint: Path,
    images: Sequence[ImageEvaluationInput],
    *,
    model_kind: str,
    input_size_px: int,
    confidence: float,
    device: str,
) -> dict[str, tuple[Box, ...]]:
    """Run square-letterbox inference and map output to native VisDrone classes."""
    try:
        ultralytics = import_module("ultralytics")
    except ImportError as error:
        raise RuntimeError("run `uv sync --group detection` for Phase 3") from error
    if model_kind not in {"native", "coco"}:
        raise ValueError("model_kind must be native or coco")
    root = Path(__file__).resolve().parents[3]
    coco_config = load_model_config(root / "configs" / "models" / "yolo26n_coco.yaml")
    if model_kind == "coco":
        checkpoint = ensure_checkpoint(checkpoint, coco_config)
    coco_mapping = coco_config.class_mapping
    model = ultralytics.YOLO(str(checkpoint))
    predictions: dict[str, tuple[Box, ...]] = {}
    batch_size = 16
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        results = model.predict(
            [str(root / image.sample.relative_image_path) for image in batch],
            imgsz=input_size_px,
            conf=confidence,
            device=device,
            rect=False,
            batch=batch_size,
            verbose=False,
        )
        for image, result in zip(batch, results, strict=True):
            names: Mapping[int, str] = result.names
            boxes: list[Box] = []
            if result.boxes is not None:
                for xyxy, score, class_id in zip(
                    result.boxes.xyxy.cpu().tolist(),
                    result.boxes.conf.cpu().tolist(),
                    result.boxes.cls.cpu().tolist(),
                    strict=True,
                ):
                    source_name = names[int(class_id)]
                    target_name = (
                        coco_mapping.get(source_name, source_name)
                        if model_kind == "coco"
                        else source_name
                    )
                    if target_name in DETECTION_ID_BY_LABEL:
                        boxes.append(
                            Box(
                                CLASS_NAMES.index(target_name),
                                tuple(float(value) for value in xyxy),  # type: ignore[arg-type]
                                float(score),
                            )
                        )
            predictions[image.sample.original_id] = tuple(boxes)
    return predictions


def train_phase3(
    *,
    checkpoint: Path,
    data_yaml: Path,
    training_config: Path,
    output_dir: Path,
    device: str,
) -> dict[str, str]:
    """Fine-tune and save SHA-256 records for distinct best and last weights."""
    config = yaml.safe_load(training_config.read_text(encoding="utf-8"))
    set_deterministic_seed(int(config["seed"]))
    try:
        ultralytics = import_module("ultralytics")
    except ImportError as error:
        raise RuntimeError("run `uv sync --group detection` for Phase 3") from error
    root = Path(__file__).resolve().parents[3]
    checkpoint = ensure_checkpoint(
        checkpoint, load_model_config(root / "configs" / "models" / "yolo26n_coco.yaml")
    )
    run_dir = output_dir / "train"
    ultralytics.YOLO(str(checkpoint)).train(
        data=str(data_yaml),
        imgsz=int(config["input_size_px"]),
        epochs=int(config["epochs"]),
        batch=int(config["batch"]),
        workers=int(config["workers"]),
        seed=int(config["seed"]),
        deterministic=bool(config["deterministic"]),
        rect=bool(config["rect"]),
        amp=bool(config["amp"]),
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=False,
        device=device,
        **{
            key: config[key]
            for key in (
                "hsv_h",
                "hsv_s",
                "hsv_v",
                "degrees",
                "translate",
                "scale",
                "shear",
                "perspective",
                "flipud",
                "fliplr",
                "mosaic",
                "mixup",
                "copy_paste",
                "erasing",
            )
        },
    )
    checkpoints = {
        "best": run_dir / "weights" / "best.pt",
        "last": run_dir / "weights" / "last.pt",
    }
    missing = [name for name, path in checkpoints.items() if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"training did not produce checkpoint(s): {', '.join(missing)}"
        )
    result = {
        name: f"sha256:{sha256(path.read_bytes()).hexdigest()}"
        for name, path in checkpoints.items()
    }
    (output_dir / "checkpoint_hashes.json").write_text(
        json.dumps(
            {
                "checkpoints": {
                    name: {"path": str(path), "sha256": result[name]}
                    for name, path in checkpoints.items()
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return result
