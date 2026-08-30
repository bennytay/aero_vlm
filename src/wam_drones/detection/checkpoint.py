"""Checked download for the pinned public checkpoint."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen

from wam_drones.detection.model_config import DetectorModelConfig, verify_checkpoint


def ensure_checkpoint(path: Path, config: DetectorModelConfig) -> Path:
    """Return a verified checkpoint, downloading atomically when absent."""
    if path.exists():
        verify_checkpoint(path, config.checkpoint_sha256)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            with urlopen(config.checkpoint_url, timeout=120) as response:
                shutil.copyfileobj(response, temporary)
            verify_checkpoint(temporary_path, config.checkpoint_sha256)
            temporary_path.replace(path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    return path
