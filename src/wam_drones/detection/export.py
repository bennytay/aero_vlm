"""Portable model export helpers."""

from importlib import import_module
from pathlib import Path
from shutil import move
from typing import Any

from wam_drones.detection.model_config import DetectorModelConfig, verify_checkpoint


def export_onnx(
    checkpoint: Path,
    config: DetectorModelConfig,
    output_dir: Path,
) -> Path:
    """Export the pinned checkpoint and require ONNX to parse the result."""
    verify_checkpoint(checkpoint, config.checkpoint_sha256)
    try:
        onnx = import_module("onnx")
        ultralytics = import_module("ultralytics")
    except ImportError as error:
        raise RuntimeError(
            "export dependencies are missing; run `uv sync --group detection`"
        ) from error
    if ultralytics.__version__ != config.package_version:
        raise RuntimeError("Ultralytics version does not match model configuration")
    output_dir.mkdir(parents=True, exist_ok=True)
    model = ultralytics.YOLO(str(checkpoint))
    exported: Any = model.export(
        format="onnx",
        imgsz=config.input_size_px,
        simplify=False,
        dynamic=False,
        device="cpu",
    )
    exported_path = Path(str(exported))
    destination = output_dir / f"{checkpoint.stem}.onnx"
    if exported_path.resolve() != destination.resolve():
        move(exported_path, destination)
    onnx.checker.check_model(onnx.load(str(destination)))
    return destination
