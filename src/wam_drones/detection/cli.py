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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_model_config(args.config)
    return int(args.handler(args, config))


if __name__ == "__main__":
    raise SystemExit(main())
