"""Atomic, hash-verified downloads shared by dataset and model tooling."""

from __future__ import annotations

import shutil
import tempfile
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen


def file_sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest for a file."""
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_with_sha256(
    url: str,
    dest: Path,
    expected_sha256: str | None = None,
    *,
    timeout: int = 120,
) -> str:
    """Return dest's SHA-256, downloading atomically first if it is absent.

    Verifies against `expected_sha256` when given; otherwise returns the
    actual digest so a first-time caller can pin it.
    """
    if dest.exists():
        actual = file_sha256(dest)
        if expected_sha256 is not None and actual != expected_sha256:
            raise ValueError(
                f"SHA-256 mismatch for {dest}: "
                f"expected {expected_sha256}, got {actual}"
            )
        return actual
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            with urlopen(url, timeout=timeout) as response:
                shutil.copyfileobj(response, temporary)
            temporary.flush()
            actual = file_sha256(temporary_path)
            if expected_sha256 is not None and actual != expected_sha256:
                raise ValueError(
                    f"SHA-256 mismatch for {url}: "
                    f"expected {expected_sha256}, got {actual}"
                )
            temporary_path.replace(dest)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    return actual
