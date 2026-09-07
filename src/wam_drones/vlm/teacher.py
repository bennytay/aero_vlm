"""Read validated, evaluation-grade tracking artefacts without mutating them."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from wam_drones.detection.contracts import FrameTracks, validate_frame_sequence

TEACHER_ARTIFACT_VERSION: Final = "v1"


class _FrozenRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FrameProvenance(_FrozenRecord):
    """The immutable source identity for one global tracker frame."""

    artifact_version: Literal["v1"] = TEACHER_ARTIFACT_VERSION
    frame_id: int = Field(ge=0)
    sequence_id: str = Field(min_length=1)
    source_frame_number: int = Field(ge=1)
    source_path: str = Field(min_length=1)
    image_width_px: int = Field(gt=0)
    image_height_px: int = Field(gt=0)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class TeacherRunMetadata(_FrozenRecord):
    """Reproducibility data that is intentionally outside ``FrameTracks``."""

    artifact_version: Literal["v1"] = TEACHER_ARTIFACT_VERSION
    artifact_kind: Literal["evaluation", "demo"]
    detector_checkpoint: str = Field(min_length=1)
    detector_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tracker_config: str = Field(min_length=1)
    tracker_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tracker_name: str = Field(min_length=1)
    tracker_thresholds: dict[str, float | int | bool]
    protocol: str | None = None
    protocol_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    code_revision: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_protocol_hash(self) -> TeacherRunMetadata:
        if (self.protocol is None) != (self.protocol_sha256 is None):
            raise ValueError("protocol and protocol_sha256 must be supplied together")
        return self


class TeacherArtifact(_FrozenRecord):
    """A read-only join of tracks, source provenance, and run metadata."""

    metadata: TeacherRunMetadata
    tracks: tuple[FrameTracks, ...]
    provenance: tuple[FrameProvenance, ...]

    @model_validator(mode="after")
    def validate_join(self) -> TeacherArtifact:
        track_ids = {frame.frame_id for frame in self.tracks}
        provenance_ids = {item.frame_id for item in self.provenance}
        if len(provenance_ids) != len(self.provenance):
            raise ValueError("provenance frame IDs must be unique")
        if track_ids != provenance_ids:
            raise ValueError("every track frame must map one-to-one to provenance")
        provenance_by_id = {item.frame_id: item for item in self.provenance}
        frames_by_sequence: dict[str, list[FrameTracks]] = {}
        for frame in self.tracks:
            frames_by_sequence.setdefault(
                provenance_by_id[frame.frame_id].sequence_id, []
            ).append(frame)
        for frames in frames_by_sequence.values():
            validate_frame_sequence(frames)
        return self

    def observations(self) -> Iterator[tuple[FrameProvenance, FrameTracks]]:
        """Yield source frames and full track state, including stale observations."""
        provenance_by_id = {item.frame_id: item for item in self.provenance}
        for tracks in self.tracks:
            yield provenance_by_id[tracks.frame_id], tracks


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of bytes used as an artefact or RGB-frame identity."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(
    path: Path, model: type[FrameTracks] | type[FrameProvenance]
) -> tuple[FrameTracks | FrameProvenance, ...]:
    records: list[FrameTracks | FrameProvenance] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            records.append(model.model_validate_json(line))
        except Exception as error:
            raise ValueError(
                f"invalid {path.name} record at line {line_number}"
            ) from error
    return tuple(records)


def load_teacher_artifact(
    tracks_path: Path,
    provenance_path: Path,
    metadata_path: Path,
) -> TeacherArtifact:
    """Load the evaluation artefact strictly; no labels or boxes are repaired."""
    return TeacherArtifact(
        metadata=TeacherRunMetadata.model_validate_json(
            metadata_path.read_text(encoding="utf-8")
        ),
        tracks=cast(tuple[FrameTracks, ...], _read_jsonl(tracks_path, FrameTracks)),
        provenance=cast(
            tuple[FrameProvenance, ...],
            _read_jsonl(provenance_path, FrameProvenance),
        ),
    )


def validate_teacher_artifact(
    tracks_path: Path,
    provenance_path: Path,
    metadata_path: Path,
) -> TeacherArtifact:
    """Load and validate a complete tracking teacher artefact."""
    return load_teacher_artifact(tracks_path, provenance_path, metadata_path)


def validate_track_jsonl(path: Path) -> tuple[FrameTracks, ...]:
    """Strictly validate a standalone tracking sidecar, such as a demo export."""
    frames = cast(tuple[FrameTracks, ...], _read_jsonl(path, FrameTracks))
    validate_frame_sequence(frames)
    return frames


def write_teacher_metadata(path: Path, metadata: TeacherRunMetadata) -> None:
    """Write validated metadata with a stable representation."""
    path.write_text(metadata.model_dump_json(indent=2) + "\n", encoding="utf-8")


def write_provenance_index(path: Path, records: list[FrameProvenance]) -> None:
    """Write one validated source mapping per track frame."""
    path.write_text(
        "".join(record.model_dump_json() + "\n" for record in records),
        encoding="utf-8",
    )
