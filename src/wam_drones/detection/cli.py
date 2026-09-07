"""Command-line smoke test for images, folders, videos, and ONNX export."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, replace
from importlib import import_module
from math import hypot
from pathlib import Path
from typing import Any, Literal

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
from wam_drones.detection.vocabulary import DetectionLabel
from wam_drones.tracking.efficiency import (
    detector_runs,
    has_confident_small_detection,
    load_efficiency_protocol,
    merge_full_frame_and_tile_detections,
    should_run_tiles,
    tile_boxes_2x2,
    tile_detection_to_full_frame,
    validate_efficiency_profile,
)
from wam_drones.tracking.offline import (
    GroundTruth,
    OfflineTracker,
    TrackingDetection,
    evaluate_tracking,
    load_tracker_config,
    mot_lines,
)
from wam_drones.vlm.teacher import (
    FrameProvenance,
    TeacherRunMetadata,
    sha256_file,
    validate_teacher_artifact,
    validate_track_jsonl,
    write_provenance_index,
    write_teacher_metadata,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _write_jsonl(path: Path, payloads: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = "\n".join(payloads)
    path.write_text(f"{contents}\n" if contents else "", encoding="utf-8")


def _repository_path(path: Path) -> str:
    """Use repository-relative paths in artefact metadata whenever possible."""
    try:
        return path.resolve().relative_to(repository_root()).as_posix()
    except ValueError:
        return str(path.resolve())


def _code_revision() -> str:
    """Record the exact revision, retaining a clear marker outside a Git checkout."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root(),
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unversioned"


def _teacher_metadata(
    *,
    artifact_kind: Literal["evaluation", "demo"],
    checkpoint: Path,
    tracker_config_path: Path,
    tracker: OfflineTracker,
    protocol: Path | None,
) -> TeacherRunMetadata:
    """Capture the detector, tracker, protocol, and code identities for a run."""
    config = tracker.config
    thresholds = {
        "high_confidence_threshold": config.high_confidence_threshold,
        "low_confidence_threshold": config.low_confidence_threshold,
        "new_track_threshold": config.new_track_threshold,
        "match_iou_threshold": config.match_iou_threshold,
        "low_match_iou_threshold": config.low_match_iou_threshold,
        "max_time_since_update_frames": config.max_time_since_update_frames,
        "min_hits": config.min_hits,
        "camera_motion_compensation": config.camera_motion_compensation,
        "reid_enabled": config.reid_enabled,
        "source_fps": config.source_fps,
    }
    return TeacherRunMetadata(
        artifact_kind=artifact_kind,
        detector_checkpoint=_repository_path(checkpoint),
        detector_checkpoint_sha256=sha256_file(checkpoint),
        tracker_config=_repository_path(tracker_config_path),
        tracker_config_sha256=sha256_file(tracker_config_path),
        tracker_name=config.name,
        tracker_thresholds=thresholds,
        protocol=_repository_path(protocol) if protocol is not None else None,
        protocol_sha256=sha256_file(protocol) if protocol is not None else None,
        code_revision=_code_revision(),
    )


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


def _phase4_protocol(path: Path) -> dict[str, tuple[str, ...]]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Phase 4 protocol must be a YAML mapping")
    development = tuple(payload.get("development_sequences", ()))
    held_out = tuple(payload.get("held_out_sequences", ()))
    if not development or not held_out or set(development) & set(held_out):
        raise ValueError("Phase 4 protocol must contain disjoint non-empty splits")
    return {"development": development, "held_out": held_out}


def run_phase4_evaluate(args: argparse.Namespace, _config: DetectorModelConfig) -> int:
    """Run a native-class checkpoint over original VisDrone-MOT frames offline."""
    try:
        cv2 = import_module("cv2")
        ultralytics = import_module("ultralytics")
    except ImportError as error:
        raise RuntimeError("run `uv sync --group detection` for Phase 4") from error
    from wam_drones.dataset.visdrone_mot import (
        VisDroneMotBox,
        parse_visdrone_mot_annotation,
    )

    protocol = _phase4_protocol(args.protocol)
    selected = protocol[args.partition]
    available = {
        path.name for path in (args.source_dir / "sequences").iterdir() if path.is_dir()
    }
    missing = set(selected) - available
    if missing:
        raise ValueError(
            f"source directory is missing protocol sequences: {sorted(missing)}"
        )
    tracker = OfflineTracker(load_tracker_config(args.tracker_config))
    model = ultralytics.YOLO(str(args.checkpoint))
    names = tuple(DetectionLabel(name) for name in model.names.values())
    if len(names) != 10:
        raise ValueError("Phase 4 checkpoint must use the ten native VisDrone classes")
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    detector_ns = tracker_ns = 0
    all_ground_truth: dict[int, tuple[GroundTruth, ...]] = {}
    all_tracks = {}
    all_camera_motion: dict[int, float] = {}
    predictions_jsonl: list[str] = []
    provenance_records: list[FrameProvenance] = []
    sequence_metrics: dict[str, object] = {}
    frame_offset = 0
    for sequence_index, sequence_id in enumerate(selected):
        sequence = args.source_dir / "sequences" / sequence_id
        annotation_path = args.source_dir / "annotations" / f"{sequence_id}.txt"
        gt_by_frame: dict[int, list[VisDroneMotBox]] = {}
        for box in parse_visdrone_mot_annotation(
            annotation_path.read_text(encoding="utf-8")
        ):
            if not box.is_trainable:
                continue
            # dimensions are read just below, avoiding rounded image-space boxes.
            gt_by_frame.setdefault(box.frame_index, []).append(box)
        sequence_frames = sorted(sequence.glob("*.jpg"))
        local_ground_truth: dict[int, tuple[GroundTruth, ...]] = {}
        local_tracks = {}
        local_camera_motion: dict[int, float] = {}
        mot_output: list[str] = []
        for local_index, image_path in enumerate(sequence_frames):
            if args.frame_limit is not None and local_index >= args.frame_limit:
                break
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"cannot decode image: {image_path}")
            height, width = image.shape[:2]
            frame_number = int(image_path.stem)
            captured_ns = int(
                (frame_number - 1) * 1_000_000_000 / tracker.config.source_fps
            )
            started = __import__("time").monotonic_ns()
            result = model.predict(
                image,
                conf=args.confidence,
                imgsz=args.input_size,
                device=args.device,
                verbose=False,
            )[0]
            detector_ns += __import__("time").monotonic_ns() - started
            detections = []
            if result.boxes is not None:
                for xyxy, confidence, class_id in zip(
                    result.boxes.xyxy.cpu().tolist(),
                    result.boxes.conf.cpu().tolist(),
                    result.boxes.cls.cpu().tolist(),
                    strict=True,
                ):
                    label = names[int(class_id)]
                    detections.append(
                        TrackingDetection(
                            label,
                            float(confidence),
                            (
                                xyxy[0] / width,
                                xyxy[1] / height,
                                xyxy[2] / width,
                                xyxy[3] / height,
                            ),
                        )
                    )
            global_frame = frame_offset + local_index
            tracker_started = __import__("time").monotonic_ns()
            tracks = tracker.update(
                detections, frame_id=global_frame, captured_at_monotonic_ns=captured_ns
            )
            tracker_ns += __import__("time").monotonic_ns() - tracker_started
            local_tracks[global_frame] = tracks
            local_camera_motion[global_frame] = tracker.last_camera_motion_norm
            predictions_jsonl.append(tracks.model_dump_json())
            provenance_records.append(
                FrameProvenance(
                    frame_id=global_frame,
                    sequence_id=sequence_id,
                    source_frame_number=frame_number,
                    source_path=image_path.relative_to(args.source_dir).as_posix(),
                    image_width_px=width,
                    image_height_px=height,
                    source_sha256=sha256_file(image_path),
                )
            )
            mot_output.extend(mot_lines((tracks,), width, height))
            local_ground_truth[global_frame] = tuple(
                GroundTruth(
                    global_frame,
                    sequence_index * 1_000_000 + item.target_id,
                    DetectionLabel(list(DetectionLabel)[item.category - 1].value),
                    (
                        item.bbox_xyxy_px[0] / width,
                        item.bbox_xyxy_px[1] / height,
                        item.bbox_xyxy_px[2] / width,
                        item.bbox_xyxy_px[3] / height,
                    ),
                    item.occlusion,
                )
                for item in gt_by_frame.get(frame_number, [])
            )
            # Tracker IDs are sequence-local by design. Offset only the
            # aggregate evaluation view so IDs from separate videos cannot
            # create false cross-sequence associations.
            all_tracks[global_frame] = tracks.model_copy(
                update={
                    "tracks": tuple(
                        track.model_copy(
                            update={
                                "track_id": sequence_index * 1_000_000 + track.track_id
                            }
                        )
                        for track in tracks.tracks
                    )
                }
            )
            all_camera_motion[global_frame] = tracker.last_camera_motion_norm
        (output_dir / f"{sequence_id}.txt").write_text(
            "\n".join(mot_output) + ("\n" if mot_output else ""), encoding="utf-8"
        )
        all_ground_truth.update(local_ground_truth)
        sequence_metrics[sequence_id] = evaluate_tracking(
            local_ground_truth,
            local_tracks,
            camera_motion_by_frame=local_camera_motion,
        )
        frame_offset += len(sequence_frames) + 1
        tracker = OfflineTracker(tracker.config)  # identities must never span videos
    (output_dir / "tracks.jsonl").write_text(
        "\n".join(predictions_jsonl) + ("\n" if predictions_jsonl else ""),
        encoding="utf-8",
    )
    provenance_path = output_dir / "frame_provenance.jsonl"
    metadata_path = output_dir / "teacher_metadata.json"
    write_provenance_index(provenance_path, provenance_records)
    write_teacher_metadata(
        metadata_path,
        _teacher_metadata(
            artifact_kind="evaluation",
            checkpoint=args.checkpoint,
            tracker_config_path=args.tracker_config,
            tracker=tracker,
            protocol=args.protocol,
        ),
    )
    validate_teacher_artifact(
        output_dir / "tracks.jsonl", provenance_path, metadata_path
    )
    metrics = evaluate_tracking(
        all_ground_truth,
        all_tracks,
        camera_motion_by_frame=all_camera_motion,
    )
    metrics["latency"] = {
        "detector_ms": detector_ns / 1_000_000,
        "tracker_ms": tracker_ns / 1_000_000,
        "frames": len(all_tracks),
        "mean_detector_ms_per_frame": detector_ns / max(len(all_tracks), 1) / 1_000_000,
        "mean_tracker_ms_per_frame": tracker_ns / max(len(all_tracks), 1) / 1_000_000,
    }
    metrics["partition"] = args.partition
    metrics["sequences"] = sequence_metrics
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    return 0


def _phase5_detections(
    model: Any,
    image: Any,
    *,
    names: tuple[DetectionLabel, ...],
    width: int,
    height: int,
    input_size: int,
    confidence: float,
    device: str,
    half: bool,
    tile: tuple[float, float, float, float] | None = None,
) -> tuple[TrackingDetection, ...]:
    """Run one backend inference and return boxes in original-frame coordinates."""
    if tile is not None:
        tx1, ty1, tx2, ty2 = tile
        x1, x2 = round(tx1 * width), round(tx2 * width)
        y1, y2 = round(ty1 * height), round(ty2 * height)
        image = image[y1:y2, x1:x2]
        height, width = image.shape[:2]
    prediction_options: dict[str, Any] = {
        "conf": confidence,
        "imgsz": input_size,
        "device": device,
        "verbose": False,
    }
    if half:
        prediction_options["half"] = True
    result = model.predict(image, **prediction_options)[0]
    detections: list[TrackingDetection] = []
    if result.boxes is not None:
        for xyxy, score, class_id in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
            strict=True,
        ):
            detection = TrackingDetection(
                names[int(class_id)],
                float(score),
                (xyxy[0] / width, xyxy[1] / height, xyxy[2] / width, xyxy[3] / height),
            )
            detections.append(
                tile_detection_to_full_frame(detection, tile)
                if tile is not None
                else detection
            )
    return tuple(detections)


def _write_phase5_preview(image: Any, tracks: Any, writer: Any) -> None:
    """Encode a deliberately simple preview inside the measured boundary."""
    cv2 = import_module("cv2")
    height, width = image.shape[:2]
    for track in tracks.tracks:
        x1, y1, x2, y2 = track.bbox_norm_xyxy
        cv2.rectangle(
            image,
            (round(x1 * width), round(y1 * height)),
            (round(x2 * width), round(y2 * height)),
            (0, 255, 0) if track.observed_this_frame else (0, 165, 255),
            1,
        )
        cv2.putText(
            image,
            str(track.track_id),
            (round(x1 * width), max(12, round(y1 * height) - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )
    writer.write(image)


def run_phase5_evaluate(args: argparse.Namespace, _config: DetectorModelConfig) -> int:
    """Measure one pre-registered Phase 5 development profile end to end."""
    try:
        cv2 = import_module("cv2")
        ultralytics = import_module("ultralytics")
    except ImportError as error:
        raise RuntimeError("run `uv sync --group detection` for Phase 5") from error
    from wam_drones.dataset.visdrone_mot import (
        VisDroneMotBox,
        parse_visdrone_mot_annotation,
    )

    protocol = load_efficiency_protocol(args.protocol)
    policy = next(
        (item for item in protocol.tiling_policies if item.mode == args.tiling_mode),
        None,
    )
    if policy is None:
        raise ValueError(
            f"tiling mode is not in the Phase 5 matrix: {args.tiling_mode}"
        )
    try:
        tracker_config = args.tracker_config.relative_to(repository_root()).as_posix()
    except ValueError:
        tracker_config = args.tracker_config.as_posix()
    validate_efficiency_profile(
        protocol,
        input_size_px=args.input_size,
        precision=args.precision,
        detector_cadence_frames=args.detector_cadence,
        tiling_policy=policy,
        tracker_config=tracker_config,
        preview=args.preview,
    )
    if args.precision == "int8":
        raise ValueError(
            "the PyTorch Phase 5 backend cannot measure INT8; use a target-native "
            "backend in Phase 6"
        )
    if args.partition == "held_out":
        if args.selection_record is None or not args.selection_record.is_file():
            raise ValueError(
                "held-out Phase 5 evaluation requires a frozen selection record"
            )
        selected_sequences = protocol.held_out_sequences
    else:
        selected_sequences = protocol.development_sequences
    expected = set(selected_sequences)
    available = {
        path.name for path in (args.source_dir / "sequences").iterdir() if path.is_dir()
    }
    missing = expected - available
    if missing:
        raise ValueError(
            f"source directory is missing development sequences: {sorted(missing)}"
        )
    model = ultralytics.YOLO(str(args.checkpoint))
    names = tuple(DetectionLabel(name) for name in model.names.values())
    if len(names) != 10:
        raise ValueError("Phase 5 checkpoint must use the ten native VisDrone classes")
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tracker = OfflineTracker(load_tracker_config(args.tracker_config))
    all_ground_truth: dict[int, tuple[GroundTruth, ...]] = {}
    all_tracks = {}
    all_camera_motion: dict[int, float] = {}
    timings: list[dict[str, str | bool | int | float]] = []
    tracks_jsonl: list[str] = []
    sequence_metrics: dict[str, object] = {}
    frame_offset = 0
    half = args.precision == "fp16"
    for sequence_index, sequence_id in enumerate(selected_sequences):
        sequence_dir = args.source_dir / "sequences" / sequence_id
        annotation_path = args.source_dir / "annotations" / f"{sequence_id}.txt"
        gt_by_frame: dict[int, list[VisDroneMotBox]] = {}
        for box in parse_visdrone_mot_annotation(
            annotation_path.read_text(encoding="utf-8")
        ):
            if box.is_trainable:
                gt_by_frame.setdefault(box.frame_index, []).append(box)
        local_ground_truth: dict[int, tuple[GroundTruth, ...]] = {}
        local_tracks = {}
        local_camera_motion: dict[int, float] = {}
        writer = None
        last_confident_small_frame: int | None = None
        mot_output: list[str] = []
        image_paths = sorted(sequence_dir.glob("*.jpg"))
        for local_index, image_path in enumerate(image_paths):
            if args.frame_limit is not None and local_index >= args.frame_limit:
                break
            pipeline_started_ns = __import__("time").monotonic_ns()
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"cannot decode image: {image_path}")
            height, width = image.shape[:2]
            frame_number = int(image_path.stem)
            captured_ns = int((frame_number - 1) * 1_000_000_000 / protocol.source_fps)
            detector_started_ns = __import__("time").monotonic_ns()
            fresh_detection = detector_runs(local_index, args.detector_cadence)
            tile_count = 0
            detections: tuple[TrackingDetection, ...] = ()
            if fresh_detection:
                full_frame = _phase5_detections(
                    model, image, names=names, width=width, height=height,
                    input_size=args.input_size, confidence=args.confidence,
                    device=args.device, half=half,
                )
                if has_confident_small_detection(full_frame, policy):
                    last_confident_small_frame = local_index
                detections = full_frame
                if should_run_tiles(
                    policy,
                    frame_index=local_index,
                    last_confident_small_detection_frame=last_confident_small_frame,
                ):
                    tile_count = 4
                    tiles = tuple(
                        detection
                        for tile in tile_boxes_2x2(policy.overlap_fraction)
                        for detection in _phase5_detections(
                            model, image, names=names, width=width, height=height,
                            input_size=args.input_size, confidence=args.confidence,
                            device=args.device, half=half, tile=tile,
                        )
                    )
                    detections = merge_full_frame_and_tile_detections(
                        (*full_frame, *tiles)
                    )
            detector_ns = __import__("time").monotonic_ns() - detector_started_ns
            tracker_started_ns = __import__("time").monotonic_ns()
            global_frame = frame_offset + local_index
            tracks = tracker.update(
                detections, frame_id=global_frame, captured_at_monotonic_ns=captured_ns
            )
            tracker_ns = __import__("time").monotonic_ns() - tracker_started_ns
            if args.preview:
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(output_dir / f"{sequence_id}_preview.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        protocol.source_fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError("failed to open Phase 5 preview writer")
                _write_phase5_preview(image, tracks, writer)
            pipeline_ns = __import__("time").monotonic_ns() - pipeline_started_ns
            timings.append({
                "sequence_id": sequence_id, "frame_number": frame_number,
                "fresh_detection": fresh_detection, "tile_count": tile_count,
                "detector_ms": detector_ns / 1_000_000,
                "tracker_ms": tracker_ns / 1_000_000,
                "pipeline_ms": pipeline_ns / 1_000_000,
            })
            local_tracks[global_frame] = tracks
            local_camera_motion[global_frame] = tracker.last_camera_motion_norm
            tracks_jsonl.append(tracks.model_dump_json())
            mot_output.extend(mot_lines((tracks,), width, height))
            local_ground_truth[global_frame] = tuple(
                GroundTruth(
                    global_frame, sequence_index * 1_000_000 + item.target_id,
                    DetectionLabel(list(DetectionLabel)[item.category - 1].value),
                    tuple(value / divisor for value, divisor in zip(
                        item.bbox_xyxy_px, (width, height, width, height), strict=True
                    )), item.occlusion,
                )
                for item in gt_by_frame.get(frame_number, [])
            )
            all_tracks[global_frame] = tracks.model_copy(
                update={
                    "tracks": tuple(
                        track.model_copy(
                            update={
                                "track_id": sequence_index * 1_000_000
                                + track.track_id
                            }
                        )
                        for track in tracks.tracks
                    )
                }
            )
            all_camera_motion[global_frame] = tracker.last_camera_motion_norm
        if writer is not None:
            writer.release()
        (output_dir / f"{sequence_id}.txt").write_text(
            "\n".join(mot_output) + ("\n" if mot_output else ""), encoding="utf-8"
        )
        all_ground_truth.update(local_ground_truth)
        sequence_metrics[sequence_id] = evaluate_tracking(
            local_ground_truth, local_tracks, camera_motion_by_frame=local_camera_motion
        )
        frame_offset += len(image_paths) + 1
        tracker = OfflineTracker(tracker.config)
    metrics = evaluate_tracking(
        all_ground_truth, all_tracks, camera_motion_by_frame=all_camera_motion
    )
    pipeline_values = [float(item["pipeline_ms"]) for item in timings]
    metrics["profile"] = {
        "input_size_px": args.input_size, "precision": args.precision,
        "detector_cadence_frames": args.detector_cadence, "tiling_mode": policy.mode,
        "tracker_config": tracker_config, "preview": args.preview,
    }
    metrics["partition"] = args.partition
    metrics["latency"] = {
        "frames": len(timings),
        "mean_pipeline_ms_per_frame": sum(pipeline_values)
        / max(len(pipeline_values), 1),
        "mean_detector_ms_per_frame": sum(
            float(item["detector_ms"]) for item in timings
        )
        / max(len(timings), 1),
        "mean_tracker_ms_per_frame": sum(
            float(item["tracker_ms"]) for item in timings
        )
        / max(len(timings), 1),
        "tile_invocations": sum(int(item["tile_count"]) for item in timings),
    }
    metrics["sequences"] = sequence_metrics
    (output_dir / "tracks.jsonl").write_text(
        "\n".join(tracks_jsonl) + "\n", encoding="utf-8"
    )
    (output_dir / "timings.jsonl").write_text(
        "\n".join(json.dumps(item) for item in timings) + "\n", encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    return 0


def _draw_demo_overlay(
    image: Any,
    tracks: Any,
    *,
    frame_index: int,
    pipeline_ms: float,
    detector_ran: bool,
) -> None:
    """Draw presentation metadata without changing tracking or evaluation output."""
    cv2 = import_module("cv2")
    height, width = image.shape[:2]
    cv2.rectangle(image, (0, 0), (width, 52), (16, 16, 16), -1)
    fps = 1_000.0 / pipeline_ms if pipeline_ms > 0 else 0.0
    status = "FRESH DETECTION" if detector_ran else "TRACK PROPAGATION"
    cv2.putText(
        image,
        f"uAeroTrack  |  {status}  |  {pipeline_ms:.1f} ms  |  {fps:.1f} FPS",
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        image,
        f"frame {frame_index}  |  native VisDrone classes  |  desktop prototype",
        (12, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (190, 190, 190),
        1,
    )
    for track in tracks.tracks:
        x1, y1, x2, y2 = track.bbox_norm_xyxy
        color = (0, 220, 0) if track.observed_this_frame else (0, 165, 255)
        left, top = round(x1 * width), round(y1 * height)
        right, bottom = round(x2 * width), round(y2 * height)
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        state = "fresh" if track.observed_this_frame else "propagated"
        label = f"{track.label.value} {track.confidence:.2f}  #{track.track_id} {state}"
        text_top = max(68, top - 6)
        cv2.putText(
            image,
            label,
            (left, text_top),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
        )


def _demo_tile_boxes(tiling_mode: str) -> tuple[tuple[float, float, float, float], ...]:
    """Return overlap-preserving tiles for an offline presentation render."""
    if tiling_mode == "2x2":
        return tile_boxes_2x2(0.20)
    if tiling_mode == "3x3":
        # Each 40%-of-frame crop overlaps its neighbour by 10% of the full
        # frame, keeping objects on a crop edge visible in both detections.
        return tuple(
            (left, top, left + 0.40, top + 0.40)
            for top in (0.0, 0.30, 0.60)
            for left in (0.0, 0.30, 0.60)
        )
    return ()


def _visible_demo_tracks(
    tracks: Any,
    history: dict[int, list[tuple[float, float]]],
    *,
    image_width_px: int,
    image_height_px: int,
    min_motion_px: float,
) -> Any:
    """Hide unconfirmed or static tracks from a presentation-only overlay."""
    visible = []
    for track in tracks.tracks:
        x1, y1, x2, y2 = track.bbox_norm_xyxy
        centre = ((x1 + x2) / 2, (y1 + y2) / 2)
        positions = history.setdefault(track.track_id, [])
        positions.append(centre)
        if len(positions) > 12:
            positions.pop(0)
        if track.hits < 3:
            continue
        if track.label in {DetectionLabel.BICYCLE, DetectionLabel.MOTOR}:
            visible.append(track)
            continue
        origin = positions[0]
        moved_px = hypot(
            (centre[0] - origin[0]) * image_width_px,
            (centre[1] - origin[1]) * image_height_px,
        )
        if moved_px >= min_motion_px:
            visible.append(track)
    return tracks.model_copy(update={"tracks": tuple(visible)})


def run_demo_video(args: argparse.Namespace, _config: DetectorModelConfig) -> int:
    """Render a shareable native-model demo from user-supplied video footage."""
    if args.source.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"demo source must be a video file: {args.source}")
    if not args.source.is_file():
        raise ValueError(f"demo source does not exist: {args.source}")
    if args.output.suffix.lower() != ".mp4":
        raise ValueError("demo output must use an .mp4 extension")
    try:
        cv2 = import_module("cv2")
        ultralytics = import_module("ultralytics")
    except ImportError as error:
        raise RuntimeError(
            "run `uv sync --group detection` for demo rendering"
        ) from error

    model = ultralytics.YOLO(str(args.checkpoint))
    names = tuple(DetectionLabel(name) for name in model.names.values())
    if len(names) != len(DetectionLabel):
        raise ValueError("demo checkpoint must use the ten native VisDrone classes")
    capture = cv2.VideoCapture(str(args.source))
    if not capture.isOpened():
        raise ValueError(f"cannot open demo source: {args.source}")
    reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_fps = reported_fps if reported_fps > 0 else 30.0
    tracker_config = replace(
        load_tracker_config(args.tracker_config), source_fps=source_fps
    )
    tracker = OfflineTracker(tracker_config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tracks_output = args.output.with_suffix(".tracks.jsonl")
    metrics_output = args.output.with_suffix(".metrics.json")
    writer = None
    frame_index = 0
    timings_ms: list[float] = []
    tracks_jsonl: list[str] = []
    presentation_tracks_jsonl: list[str] = []
    track_history: dict[int, list[tuple[float, float]]] = {}
    try:
        while args.frame_limit is None or frame_index < args.frame_limit:
            ok, image = capture.read()
            if not ok:
                break
            height, width = image.shape[:2]
            started_ns = __import__("time").monotonic_ns()
            detector_ran = detector_runs(frame_index, args.detector_cadence)
            detections: tuple[TrackingDetection, ...] = ()
            if detector_ran:
                full_frame = _phase5_detections(
                    model,
                    image,
                    names=names,
                    width=width,
                    height=height,
                    input_size=args.input_size,
                    confidence=args.confidence,
                    device=args.device,
                    half=False,
                )
                if args.tiling_mode != "off":
                    tile_detections = tuple(
                        detection
                        for tile in _demo_tile_boxes(args.tiling_mode)
                        for detection in _phase5_detections(
                            model,
                            image,
                            names=names,
                            width=width,
                            height=height,
                            input_size=args.input_size,
                            confidence=args.confidence,
                            device=args.device,
                            half=False,
                            tile=tile,
                        )
                    )
                    detections = merge_full_frame_and_tile_detections(
                        (*full_frame, *tile_detections)
                    )
                else:
                    detections = full_frame
            captured_ns = round(frame_index * 1_000_000_000 / source_fps)
            tracks = tracker.update(
                detections,
                frame_id=frame_index,
                captured_at_monotonic_ns=captured_ns,
            )
            visible_tracks = _visible_demo_tracks(
                tracks,
                track_history,
                image_width_px=width,
                image_height_px=height,
                min_motion_px=args.min_confirmed_motion_px,
            )
            pipeline_ms = (
                __import__("time").monotonic_ns() - started_ns
            ) / 1_000_000
            _draw_demo_overlay(
                image,
                visible_tracks,
                frame_index=frame_index,
                pipeline_ms=pipeline_ms,
                detector_ran=detector_ran,
            )
            if writer is None:
                writer = cv2.VideoWriter(
                    str(args.output),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    source_fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"cannot open demo output: {args.output}")
            writer.write(image)
            timings_ms.append(pipeline_ms)
            # The sidecar is teacher-grade tracker output.  The overlay uses a
            # presentation-only visibility filter and is saved separately.
            tracks_jsonl.append(tracks.model_dump_json())
            presentation_tracks_jsonl.append(visible_tracks.model_dump_json())
            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if not timings_ms:
        raise ValueError(f"demo source contains no frames: {args.source}")
    tracks_output.write_text("\n".join(tracks_jsonl) + "\n", encoding="utf-8")
    validate_track_jsonl(tracks_output)
    args.output.with_suffix(".presentation_tracks.jsonl").write_text(
        "\n".join(presentation_tracks_jsonl) + "\n", encoding="utf-8"
    )
    metrics_output.write_text(
        json.dumps(
            {
                "source": str(args.source),
                "frames": frame_index,
                "source_fps": source_fps,
                "mean_pipeline_ms": sum(timings_ms) / len(timings_ms),
                "p95_pipeline_ms": sorted(timings_ms)[
                    max(0, round(0.95 * len(timings_ms)) - 1)
                ],
                "detector_cadence_frames": args.detector_cadence,
                "tiling_mode": args.tiling_mode,
                "input_size_px": args.input_size,
                "confidence": args.confidence,
                "min_confirmed_motion_px": args.min_confirmed_motion_px,
                "tracker_config": str(args.tracker_config),
                "tracks_sidecar": "unfiltered FrameTracks v1",
                "presentation_tracks_sidecar": "visibility-filtered overlay tracks",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output)
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

    phase4 = subparsers.add_parser(
        "phase4-evaluate",
        help="run offline tracker on the sequence-separated VisDrone-MOT protocol",
    )
    phase4.add_argument("--checkpoint", type=Path, required=True)
    phase4.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="extracted VisDrone2019-MOT-val directory",
    )
    phase4.add_argument(
        "--tracker-config",
        type=Path,
        default=root / "configs" / "tracking" / "botsort_phase4.yaml",
    )
    phase4.add_argument(
        "--protocol",
        type=Path,
        default=root / "configs" / "tracking" / "phase4_protocol.yaml",
    )
    phase4.add_argument(
        "--partition", choices=["development", "held_out"], required=True
    )
    phase4.add_argument("--output-dir", type=Path, required=True)
    phase4.add_argument("--input-size", type=int, default=640)
    phase4.add_argument("--confidence", type=float, default=0.001)
    phase4.add_argument("--device", default="cpu")
    phase4.add_argument("--frame-limit", type=int)
    phase4.set_defaults(handler=run_phase4_evaluate)

    phase5 = subparsers.add_parser(
        "phase5-evaluate",
        help="measure one registered Phase 5 development profile end to end",
    )
    phase5.add_argument(
        "--checkpoint",
        type=Path,
        default=root
        / "evaluation"
        / "experiments"
        / "exp_20260831_phase3_visdrone_det"
        / "artefacts"
        / "train"
        / "weights"
        / "best.pt",
    )
    phase5.add_argument("--source-dir", type=Path, required=True)
    phase5.add_argument("--output-dir", type=Path, required=True)
    phase5.add_argument(
        "--partition", choices=["development", "held_out"], default="development"
    )
    phase5.add_argument(
        "--selection-record",
        type=Path,
        help="required frozen selection record when evaluating held-out sequences",
    )
    phase5.add_argument(
        "--protocol",
        type=Path,
        default=root / "configs" / "experiments" / "phase5_efficiency.yaml",
    )
    phase5.add_argument(
        "--tracker-config",
        type=Path,
        default=root / "configs" / "tracking" / "botsort_phase4.yaml",
    )
    phase5.add_argument("--input-size", type=int, default=640)
    phase5.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp32")
    phase5.add_argument("--detector-cadence", type=int, default=1)
    phase5.add_argument(
        "--tiling-mode",
        choices=["off", "scheduled_2x2", "uncertainty_2x2"],
        default="off",
    )
    phase5.add_argument(
        "--preview", action=argparse.BooleanOptionalAction, default=False
    )
    phase5.add_argument("--confidence", type=float, default=0.001)
    phase5.add_argument("--device", default="cpu")
    phase5.add_argument("--frame-limit", type=int)
    phase5.set_defaults(handler=run_phase5_evaluate)

    demo = subparsers.add_parser(
        "demo-video",
        help="render native detector and tracker overlays on a personal video",
    )
    demo.add_argument("source", type=Path, help="input MP4, MOV, AVI, MKV, or WebM")
    demo.add_argument("--output", type=Path, required=True, help="annotated MP4")
    demo.add_argument(
        "--checkpoint",
        type=Path,
        default=root
        / "evaluation"
        / "experiments"
        / "exp_20260831_phase3_visdrone_det"
        / "artefacts"
        / "train"
        / "weights"
        / "best.pt",
    )
    demo.add_argument(
        "--tracker-config",
        type=Path,
        default=root / "configs" / "tracking" / "botsort_phase4.yaml",
    )
    demo.add_argument("--input-size", type=int, default=640)
    demo.add_argument("--confidence", type=float, default=0.25)
    demo.add_argument("--detector-cadence", type=int, default=1)
    demo.add_argument(
        "--min-confirmed-motion-px",
        type=float,
        default=12.0,
        help="hide tracks until they have moved this far in the source frame",
    )
    demo.add_argument(
        "--tiling-mode",
        choices=["off", "2x2", "3x3"],
        default="off",
        help="run overlapping crops as well as full-frame inference",
    )
    demo.add_argument("--device", default="cpu")
    demo.add_argument("--frame-limit", type=int)
    demo.set_defaults(handler=run_demo_video)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_model_config(args.config)
    return int(args.handler(args, config))


if __name__ == "__main__":
    raise SystemExit(main())
