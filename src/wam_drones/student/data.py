"""Deterministic manifest/teacher-embedding preparation for distillation."""

import gzip
import json
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DistillationSample:
    """One licensed scene paired with its frozen teacher image embedding."""

    image_path: Path
    source_id: str
    expected_label: str | None
    teacher_embedding: np.ndarray[Any, np.dtype[np.float32]]


def load_photo_embeddings(path: Path) -> dict[str, np.ndarray[Any, Any]]:
    """Load only the default photo-template image embeddings."""
    result: dict[str, np.ndarray[Any, Any]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["template"] == "photo":
                result[str(row["source_id"])] = np.asarray(
                    row["embedding"], dtype=np.float32
                )
    return result


def load_samples(
    repo_root: Path, manifest_path: Path, embedding_path: Path
) -> list[DistillationSample]:
    """Join the checked manifest to cached teacher evidence by source id."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    embeddings = load_photo_embeddings(embedding_path)
    samples = []
    for row in manifest["samples"]:
        source_id = str(row["source_id"])
        if source_id not in embeddings:
            continue
        samples.append(
            DistillationSample(
                image_path=repo_root / row["image_path"],
                source_id=source_id,
                expected_label=None if row["no_target"] else row["target_label"],
                teacher_embedding=embeddings[source_id],
            )
        )
    return samples


def deterministic_split(
    samples: list[DistillationSample], seed: int, held_out_fraction: float
) -> tuple[list[DistillationSample], list[DistillationSample]]:
    """Stratify by ground truth while assigning whole source/scene ids."""
    groups: dict[str, list[DistillationSample]] = defaultdict(list)
    for sample in samples:
        groups[sample.expected_label or "no_target"].append(sample)
    train: list[DistillationSample] = []
    held_out: list[DistillationSample] = []
    for group in groups.values():
        ordered = sorted(
            group,
            key=lambda sample: sha256(f"{seed}:{sample.source_id}".encode()).digest(),
        )
        count = max(1, round(len(ordered) * held_out_fraction))
        held_out.extend(ordered[:count])
        train.extend(ordered[count:])
    return train, held_out


def projection_basis(
    samples: list[DistillationSample], dimension: int
) -> np.ndarray[Any, Any]:
    """Fit an uncentred SVD basis on train scenes only."""
    matrix = np.stack([sample.teacher_embedding for sample in samples])
    _, _, right = np.linalg.svd(matrix, full_matrices=False)
    return right[:dimension].T.astype(np.float32)


def project_normalized(
    embeddings: np.ndarray[Any, Any], basis: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    """Project teacher-space rows into the student's unit sphere."""
    projected = embeddings @ basis
    norms = np.linalg.norm(projected, axis=-1, keepdims=True).clip(min=1e-12)
    return np.asarray(projected / norms, dtype=np.float32)
