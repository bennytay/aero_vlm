"""Command-line smoke test for images, folders, videos, and ONNX export."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from importlib import import_module
from pathlib import Path

from wam_drones.detection.checkpoint import ensure_checkpoint
from wam_drones.detection.export import export_onnx
from wam_drones.detection.inference import (
    UltralyticsBackend,
    detect_image,
    detect_video,
    iter_image_folder,
)
from wam_drones.detection.model_config import DetectorModelConfig, load_model_config
from wam_drones.detection.parity import compare_frames
from wam_drones.detection.phase3 import (
    evaluate_predictions,
    infer_predictions,
    load_evaluation_inputs,
    prepare_phase3_dataset,
    train_phase3,
    write_manual_review_pack,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _write_jsonl(path: Path, payloads: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = "\n".join(payloads)
    path.write_text(f"{contents}\n" if contents else "", encoding="utf-8")


def _write_image(path: Path, image: object) -> None:
    try:
        cv2 = import_module("cv2")
    except ImportError as error:
        raise RuntimeError(
            "run `uv sync --group detection` for image output"
        ) from error
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write image: {path}")


def _run_video(
    source: Path,
    output_dir: Path,
    backend: UltralyticsBackend,
    config: DetectorModelConfig,
    frame_limit: int | None,
    track: bool,
    error_frame_count: int,
) -> None:
    try:
        cv2 = import_module("cv2")
    except ImportError as error:
        raise RuntimeError(
            "run `uv sync --group detection` for video output"
        ) from error
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    json_lines: list[str] = []
    latency_ms: list[float] = []
    error_candidates: list[tuple[float, int, object]] = []
    output_video = output_dir / "annotated.mp4"
    try:
        for frame, annotated, fps in detect_video(
            source,
            backend,
            config.class_mapping,
            track=track,
            frame_limit=frame_limit,
        ):
            json_lines.append(frame.model_dump_json())
            latency_ms.append(
                (
                    frame.produced_at_monotonic_ns
                    - frame.inference_started_at_monotonic_ns
                )
                / 1_000_000
            )
            if annotated is not None:
                highest_confidence = max(
                    (detection.confidence for detection in frame.detections),
                    default=0.0,
                )
                error_candidates.append(
                    (highest_confidence, frame.frame_id, annotated.copy())
                )
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(output_video),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (frame.image_width_px, frame.image_height_px),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(
                            f"failed to open video writer: {output_video}"
                        )
                writer.write(annotated)
    finally:
        if writer is not None:
            writer.release()
    for _, frame_id, image in sorted(error_candidates)[:error_frame_count]:
        _write_image(output_dir / "error_frames" / f"frame_{frame_id:06d}.jpg", image)
    _write_jsonl(output_dir / "predictions.jsonl", json_lines)
    _write_metrics(output_dir, latency_ms, len(json_lines))


def _write_metrics(output_dir: Path, latency_ms: list[float], frames: int) -> None:
    metrics = {
        "frames": frames,
        "mean_pipeline_latency_ms": (
            sum(latency_ms) / len(latency_ms) if latency_ms else None
        ),
        "min_pipeline_latency_ms": min(latency_ms, default=None),
        "max_pipeline_latency_ms": max(latency_ms, default=None),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )


def run_source(args: argparse.Namespace, config: DetectorModelConfig) -> int:
    checkpoint = ensure_checkpoint(args.checkpoint, config)
    backend = UltralyticsBackend(
        checkpoint, config, confidence=args.confidence, device=args.device
    )
    source: Path = args.source
    output_dir: Path = args.output_dir
    if source.is_dir():
        json_lines: list[str] = []
        latency_ms: list[float] = []
        for path, frame, annotated in iter_image_folder(
            source, backend, config.class_mapping
        ):
            json_lines.append(frame.model_dump_json())
            latency_ms.append(
                (
                    frame.produced_at_monotonic_ns
                    - frame.inference_started_at_monotonic_ns
                )
                / 1_000_000
            )
            if annotated is not None:
                _write_image(output_dir / "annotated" / path.name, annotated)
        _write_jsonl(output_dir / "predictions.jsonl", json_lines)
        _write_metrics(output_dir, latency_ms, len(json_lines))
    elif source.suffix.lower() in VIDEO_EXTENSIONS:
        _run_video(
            source,
            output_dir,
            backend,
            config,
            args.frame_limit,
            args.track,
            args.error_frame_count,
        )
    elif source.suffix.lower() in IMAGE_EXTENSIONS:
        frame, annotated = detect_image(source, backend, config.class_mapping)
        _write_jsonl(output_dir / "predictions.jsonl", [frame.model_dump_json()])
        latency = (
            frame.produced_at_monotonic_ns - frame.inference_started_at_monotonic_ns
        ) / 1_000_000
        _write_metrics(output_dir, [latency], 1)
        if annotated is not None:
            _write_image(output_dir / "annotated" / source.name, annotated)
    else:
        raise ValueError(f"unsupported source: {source}")
    return 0


def run_export(args: argparse.Namespace, config: DetectorModelConfig) -> int:
    checkpoint = ensure_checkpoint(args.checkpoint, config)
    exported = export_onnx(checkpoint, config, args.output_dir)
    print(exported)
    return 0


def run_parity(args: argparse.Namespace, config: DetectorModelConfig) -> int:
    checkpoint = ensure_checkpoint(args.checkpoint, config)
    pytorch = UltralyticsBackend(checkpoint, config, device=args.device)
    onnx = UltralyticsBackend(
        args.onnx,
        config,
        device=args.device,
        verify_hash=False,
        model_name=f"{config.name}-onnx",
    )
    reference, _ = detect_image(args.source, pytorch, config.class_mapping)
    candidate, _ = detect_image(args.source, onnx, config.class_mapping)
    report = compare_frames(reference, candidate, iou_threshold=args.iou_threshold)
    payload = asdict(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload))
    return 0


def run_phase3_prepare(args: argparse.Namespace, _config: DetectorModelConfig) -> int:
    metadata = prepare_phase3_dataset(
        repo_root=args.repo_root,
        output_dir=args.output_dir,
        train_manifest_path=args.train_manifest,
        val_manifest_path=args.val_manifest,
        dedup_report_path=args.dedup_report,
    )
    print(json.dumps(metadata, indent=2))
    return 0


def _write_phase3_report(output_dir: Path, metrics: dict[str, object]) -> None:
    overall = metrics["overall"]
    assert isinstance(overall, dict)
    lines = [
        "# Phase 3 evaluation",
        "",
        "The primary set is the official DET validation split with the known "
        "DET train/val near-duplicate evaluation image excluded. Metrics use "
        "aspect-ratio-preserving 640px letterboxing.",
        "",
        "## Overall",
        "",
        f"- mAP50-95: {overall['map50_95']}",
        f"- AP50: {overall['ap50']}",
        f"- Precision: {overall['precision']}",
        f"- Recall: {overall['recall']}",
        f"- False positives/frame: {overall['false_positives_per_frame']}",
        "",
        "The full per-class AP-small, precision, recall, false-positive/frame, "
        "and size/density/occlusion/scene breakdowns are in metrics.json.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase3_evaluate(args: argparse.Namespace, _config: DetectorModelConfig) -> int:
    images = load_evaluation_inputs(args.manifest, args.repo_root)
    predictions = infer_predictions(
        args.checkpoint,
        images,
        model_kind=args.model_kind,
        input_size_px=args.input_size,
        confidence=args.confidence,
        device=args.device,
    )
    metrics = evaluate_predictions(images, predictions, input_size_px=args.input_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    review = write_manual_review_pack(
        images,
        predictions,
        repo_root=args.repo_root,
        output_dir=args.output_dir / "manual_review",
        count=args.review_count,
    )
    metrics["manual_review"] = review
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    _write_phase3_report(args.output_dir, metrics)
    print(json.dumps(metrics["overall"], indent=2))
    return 0


def run_phase3_train(args: argparse.Namespace, _config: DetectorModelConfig) -> int:
    hashes = train_phase3(
        checkpoint=args.checkpoint,
        data_yaml=args.data_yaml,
        training_config=args.training_config,
        output_dir=args.output_dir,
        device=args.device,
    )
    print(json.dumps(hashes, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    root = repository_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs" / "models" / "yolo26n_coco.yaml",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="detect an image, folder, or video")
    run.add_argument("source", type=Path)
    run.add_argument("--checkpoint", type=Path, default=root / "models" / "yolo26n.pt")
    run.add_argument("--output-dir", type=Path, default=root / "runs" / "phase1")
    run.add_argument("--confidence", type=float, default=0.25)
    run.add_argument("--device", default="cpu")
    run.add_argument("--frame-limit", type=int)
    run.add_argument("--error-frame-count", type=int, default=3)
    run.add_argument("--track", action=argparse.BooleanOptionalAction, default=True)
    run.set_defaults(handler=run_source)

    export = subparsers.add_parser("export", help="export and validate ONNX")
    export.add_argument(
        "--checkpoint", type=Path, default=root / "models" / "yolo26n.pt"
    )
    export.add_argument("--output-dir", type=Path, default=root / "models")
    export.set_defaults(handler=run_export)

    parity = subparsers.add_parser("parity", help="compare PyTorch and ONNX boxes")
    parity.add_argument("source", type=Path)
    parity.add_argument(
        "--checkpoint", type=Path, default=root / "models" / "yolo26n.pt"
    )
    parity.add_argument("--onnx", type=Path, required=True)
    parity.add_argument(
        "--output", type=Path, default=root / "runs" / "phase1" / "parity.json"
    )
    parity.add_argument("--device", default="cpu")
    parity.add_argument("--iou-threshold", type=float, default=0.5)
    parity.set_defaults(handler=run_parity)

    phase3_prepare = subparsers.add_parser(
        "phase3-prepare",
        help="write leakage-aware VisDrone DET lists without changing raw data",
    )
    phase3_prepare.add_argument("--repo-root", type=Path, default=root)
    phase3_prepare.add_argument(
        "--train-manifest",
        type=Path,
        default=root / "data" / "manifests" / "visdrone_det_train_v1.json",
    )
    phase3_prepare.add_argument(
        "--val-manifest",
        type=Path,
        default=root / "data" / "manifests" / "visdrone_det_val_v1.json",
    )
    phase3_prepare.add_argument(
        "--dedup-report",
        type=Path,
        default=root
        / "evaluation"
        / "experiments"
        / "exp_20260831_phase2_visdrone_det"
        / "artefacts"
        / "dedup_report.json",
    )
    phase3_prepare.add_argument(
        "--output-dir",
        type=Path,
        default=root
        / "evaluation"
        / "experiments"
        / "exp_20260831_phase3_visdrone_det"
        / "artefacts"
        / "dataset",
    )
    phase3_prepare.set_defaults(handler=run_phase3_prepare)

    phase3_evaluate = subparsers.add_parser(
        "phase3-evaluate",
        help="score a COCO or native checkpoint and generate the 25+25 review pack",
    )
    phase3_evaluate.add_argument("--repo-root", type=Path, default=root)
    phase3_evaluate.add_argument("--checkpoint", type=Path, required=True)
    phase3_evaluate.add_argument(
        "--model-kind", choices=["coco", "native"], required=True
    )
    phase3_evaluate.add_argument(
        "--manifest",
        type=Path,
        default=root
        / "evaluation"
        / "experiments"
        / "exp_20260831_phase3_visdrone_det"
        / "artefacts"
        / "dataset"
        / "sequence_safe_val_manifest.json",
    )
    phase3_evaluate.add_argument("--input-size", type=int, default=640)
    phase3_evaluate.add_argument("--confidence", type=float, default=0.001)
    phase3_evaluate.add_argument("--device", default="cpu")
    phase3_evaluate.add_argument("--review-count", type=int, default=25)
    phase3_evaluate.add_argument("--output-dir", type=Path, required=True)
    phase3_evaluate.set_defaults(handler=run_phase3_evaluate)

    phase3_train = subparsers.add_parser(
        "phase3-train", help="fine-tune the pinned model with the prepared lists"
    )
    phase3_train.add_argument(
        "--checkpoint", type=Path, default=root / "models" / "yolo26n.pt"
    )
    phase3_train.add_argument(
        "--data-yaml",
        type=Path,
        default=root
        / "evaluation"
        / "experiments"
        / "exp_20260831_phase3_visdrone_det"
        / "artefacts"
        / "dataset"
        / "visdrone_det_phase3.yaml",
    )
    phase3_train.add_argument(
        "--training-config",
        type=Path,
        default=root / "configs" / "training" / "visdrone_det_phase3.yaml",
    )
    phase3_train.add_argument(
        "--output-dir",
        type=Path,
        default=root
        / "evaluation"
        / "experiments"
        / "exp_20260831_phase3_visdrone_det"
        / "artefacts",
    )
    phase3_train.add_argument("--device", default="cpu")
    phase3_train.set_defaults(handler=run_phase3_train)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_model_config(args.config)
    return int(args.handler(args, config))


if __name__ == "__main__":
    raise SystemExit(main())
