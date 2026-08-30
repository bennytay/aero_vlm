"""Checked download for the pinned public checkpoint."""

from __future__ import annotations

from pathlib import Path

from wam_drones.detection.model_config import DetectorModelConfig
from wam_drones.net import download_with_sha256


def ensure_checkpoint(path: Path, config: DetectorModelConfig) -> Path:
    """Return a verified checkpoint, downloading atomically when absent."""
    download_with_sha256(config.checkpoint_url, path, config.checkpoint_sha256)
    return path
