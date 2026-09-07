"""Locked VLM sequence split and held-out leakage guard (Step 3, MVP scope).

Full Step 3 (300-500 human-audited prompts, perceptual-hash dedup across
DET/MOT/external sources, formal leakage report) is deferred until real VLM
training data is being assembled. This module only preserves the Phase 4
tracking split and gives future data-factory code (Step 4+) a single place
to check a candidate sequence ID against it before it can leak into a
training or smoke-suite sample.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

LOCKED_SPLIT_VERSION: Final = "vlm_locked_split_v1"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_locked_split_path() -> Path:
    return repository_root() / "configs/vlm/data/locked_split_v1.json"


class LockedSplit(BaseModel):
    """The sequence-level train/held-out boundary the VLM must respect."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    source: str = Field(min_length=1)
    development_sequences: tuple[str, ...] = Field(min_length=1)
    held_out_sequences: tuple[str, ...] = Field(min_length=1)
    note: str = Field(min_length=1)


def load_locked_split(path: Path | None = None) -> LockedSplit:
    """Load the checked-in split without inferring or repairing its content."""
    resolved = path or default_locked_split_path()
    split = LockedSplit.model_validate_json(resolved.read_text(encoding="utf-8"))
    if split.schema_version != LOCKED_SPLIT_VERSION:
        raise ValueError(
            f"unsupported locked split schema_version: {split.schema_version!r}"
        )
    overlap = set(split.development_sequences) & set(split.held_out_sequences)
    if overlap:
        raise ValueError(f"sequence listed in both splits: {sorted(overlap)}")
    return split


def assert_no_held_out_leakage(
    candidate_sequence_ids: Iterable[str], split: LockedSplit | None = None
) -> None:
    """Raise if any candidate sequence is reserved for final VLM evaluation.

    Call this before sampling frames for VLM training, smoke-suite curation,
    or prompt authoring.
    """
    locked = split or load_locked_split()
    leaked = sorted(set(candidate_sequence_ids) & set(locked.held_out_sequences))
    if leaked:
        raise ValueError(
            "candidate sequences are reserved for held-out VLM evaluation and "
            f"must not be used for training or prompt authoring: {leaked}"
        )
