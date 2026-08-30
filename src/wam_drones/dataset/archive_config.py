"""Pinned identity for reproducibly downloadable dataset archives."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ArchiveSplitConfig(BaseModel):
    """One split's source URL and, once known, its pinned SHA-256."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    url: str = Field(min_length=1)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class DatasetArchiveConfig(BaseModel):
    """Versioned, licensed source for one dataset's downloadable archives."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["v1"]
    dataset: str = Field(min_length=1)
    license: str = Field(min_length=1)
    citation: str = Field(min_length=1)
    splits: dict[str, ArchiveSplitConfig]


def load_archive_config(path: Path) -> DatasetArchiveConfig:
    """Load a checked-in dataset archive definition."""
    return DatasetArchiveConfig.model_validate(
        yaml.safe_load(path.read_text(encoding="utf-8"))
    )


def pin_split_sha256(path: Path, split: str, sha256_hex: str) -> None:
    """Rewrite one split's pinned SHA-256 in place, preserving key order."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["splits"][split]["sha256"] = sha256_hex
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
