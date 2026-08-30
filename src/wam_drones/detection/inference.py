"""Portable public-detector inference and contract conversion."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol

from wam_drones.detection.contracts import Detection, FrameDetections
from wam_drones.detection.model_config import DetectorModelConfig, verify_checkpoint
from wam_drones.detection.vocabulary import DETECTION_ID_BY_LABEL, DetectionLabel


@dataclass(frozen=True)
class RawDetection:
    """Backend-neutral pixel-space detection."""

    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class ModelPrediction:
    """One backend result plus an optional annotated BGR frame."""

    image_width_px: int
    image_height_px: int
    detections: tuple[RawDetection, ...]
    annotated_bgr: Any | None = None


class DetectorBackend(Protocol):
    """Minimal backend boundary used by image, folder, and video runners."""

    model_name: str

    def predict(self, image: Any, *, track: bool = False) -> ModelPrediction:
        """Run inference on a path or decoded BGR frame."""


class UltralyticsBackend:
    """Lazy Ultralytics adapter; importing the core package stays lightweight."""

    def __init__(
        self,
        checkpoint: Path,
        config: DetectorModelConfig,
        *,
        confidence: float = 0.25,
        device: str = "cpu",
        verify_hash: bool = True,
        model_name: str | None = None,
    ) -> None:
        if verify_hash:
            verify_checkpoint(checkpoint, config.checkpoint_sha256)
        try:
            ultralytics = import_module("ultralytics")
        except ImportError as error:
            raise RuntimeError(
                "detector dependencies are missing; run `uv sync --group detection`"
            ) from error
        if ultralytics.__version__ != config.package_version:
            raise RuntimeError(
                "Ultralytics version mismatch: expected "
                f"{config.package_version}, got {ultralytics.__version__}"
            )
        self.model_name = model_name or config.name
        self._model = ultralytics.YOLO(str(checkpoint))
        self._confidence = confidence
        self._device = device
        self._input_size = config.input_size_px

    def predict(self, image: Any, *, track: bool = False) -> ModelPrediction:
        kwargs = {
            "source": image,
            "conf": self._confidence,
            "imgsz": self._input_size,
            "device": self._device,
            "verbose": False,
        }
        if track:
            results = self._model.track(
                **kwargs, persist=True, tracker="bytetrack.yaml"
            )
        else:
            results = self._model.predict(**kwargs)
        result = results[0]
        height, width = (int(value) for value in result.orig_shape)
        names: Mapping[int, str] = result.names
        raw: list[RawDetection] = []
        if result.boxes is not None:
            coordinates = result.boxes.xyxy.cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()
            classes = result.boxes.cls.cpu().tolist()
            for box, confidence, class_id in zip(
                coordinates, confidences, classes, strict=True
            ):
                raw.append(
                    RawDetection(
                        label=names[int(class_id)],
                        confidence=float(confidence),
                        bbox_xyxy=tuple(float(value) for value in box),  # type: ignore[arg-type]
                    )
                )
        return ModelPrediction(width, height, tuple(raw), result.plot())


def _normalise_box(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float] | None:
    x_min, y_min, x_max, y_max = box
    normalised = (
        min(1.0, max(0.0, x_min / width)),
        min(1.0, max(0.0, y_min / height)),
        min(1.0, max(0.0, x_max / width)),
        min(1.0, max(0.0, y_max / height)),
    )
    if normalised[0] >= normalised[2] or normalised[1] >= normalised[3]:
        return None
    return normalised


def prediction_to_contract(
    prediction: ModelPrediction,
    *,
    frame_id: int,
    captured_at_monotonic_ns: int,
    inference_started_at_monotonic_ns: int,
    produced_at_monotonic_ns: int,
    model_name: str,
    class_mapping: Mapping[str, DetectionLabel],
) -> FrameDetections:
    """Project supported COCO detections into the frozen VisDrone contract."""
    detections: list[Detection] = []
    for raw in prediction.detections:
        target_label = class_mapping.get(raw.label)
        if target_label is None:
            continue
        box = _normalise_box(
            raw.bbox_xyxy,
            prediction.image_width_px,
            prediction.image_height_px,
        )
        if box is None:
            continue
        detections.append(
            Detection(
                frame_id=frame_id,
                captured_at_monotonic_ns=captured_at_monotonic_ns,
                label_id=DETECTION_ID_BY_LABEL[target_label],
                label=target_label,
                confidence=raw.confidence,
                bbox_norm_xyxy=box,
                model_name=model_name,
            )
        )
    return FrameDetections(
        frame_id=frame_id,
        captured_at_monotonic_ns=captured_at_monotonic_ns,
        inference_started_at_monotonic_ns=inference_started_at_monotonic_ns,
        produced_at_monotonic_ns=produced_at_monotonic_ns,
        image_width_px=prediction.image_width_px,
        image_height_px=prediction.image_height_px,
        detections=tuple(detections),
    )


def detect_image(
    image: Any,
    backend: DetectorBackend,
    class_mapping: Mapping[str, DetectionLabel],
    *,
    frame_id: int = 0,
    captured_at_monotonic_ns: int | None = None,
    track: bool = False,
) -> tuple[FrameDetections, Any | None]:
    """Detect one image and return the project contract and rendered frame."""
    captured = (
        time.monotonic_ns()
        if captured_at_monotonic_ns is None
        else captured_at_monotonic_ns
    )
    started = max(captured, time.monotonic_ns())
    prediction = backend.predict(image, track=track)
    produced = max(started, time.monotonic_ns())
    return (
        prediction_to_contract(
            prediction,
            frame_id=frame_id,
            captured_at_monotonic_ns=captured,
            inference_started_at_monotonic_ns=started,
            produced_at_monotonic_ns=produced,
            model_name=backend.model_name,
            class_mapping=class_mapping,
        ),
        prediction.annotated_bgr,
    )


def iter_image_folder(
    folder: Path,
    backend: DetectorBackend,
    class_mapping: Mapping[str, DetectionLabel],
) -> Iterator[tuple[Path, FrameDetections, Any | None]]:
    """Detect supported images in stable lexical order."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    sources = sorted(
        path for path in folder.iterdir() if path.suffix.lower() in extensions
    )
    for frame_id, path in enumerate(sources):
        frame, annotated = detect_image(path, backend, class_mapping, frame_id=frame_id)
        yield path, frame, annotated


def detect_video(
    source: Path,
    backend: DetectorBackend,
    class_mapping: Mapping[str, DetectionLabel],
    *,
    track: bool = True,
    frame_limit: int | None = None,
) -> Iterator[tuple[FrameDetections, Any | None, float]]:
    """Stream a video without loading all frames into memory."""
    try:
        cv2 = import_module("cv2")
    except ImportError as error:
        raise RuntimeError(
            "video dependencies are missing; run `uv sync --group detection`"
        ) from error
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {source}")
    reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps = reported_fps if math.isfinite(reported_fps) and reported_fps > 0 else 30.0
    try:
        frame_id = 0
        while frame_limit is None or frame_id < frame_limit:
            ok, image = capture.read()
            if not ok:
                break
            frame, annotated = detect_image(
                image,
                backend,
                class_mapping,
                frame_id=frame_id,
                track=track,
            )
            yield frame, annotated, fps
            frame_id += 1
    finally:
        capture.release()


def write_jsonl(frames: Iterable[FrameDetections], path: Path) -> None:
    """Write one validated frame contract per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for frame in frames:
            handle.write(
                json.dumps(frame.model_dump(mode="json"), separators=(",", ":"))
            )
            handle.write("\n")
