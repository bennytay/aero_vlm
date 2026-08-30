"""Perceptual-hash duplicate detection across one or more manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from wam_drones.dataset.manifest import DatasetManifest, ImageRecord

HASH_SIZE = 8
DEFAULT_HAMMING_THRESHOLD = 4


def image_dhash(path: Path) -> int:
    """Return a 64-bit difference hash: robust to re-encoding, not to crops."""
    with Image.open(path) as image:
        grayscale = image.convert("L").resize(
            (HASH_SIZE + 1, HASH_SIZE), Image.Resampling.LANCZOS
        )
        pixels = list(grayscale.getdata())
    bits = 0
    for row in range(HASH_SIZE):
        offset = row * (HASH_SIZE + 1)
        for col in range(HASH_SIZE):
            bits <<= 1
            if pixels[offset + col] > pixels[offset + col + 1]:
                bits |= 1
    return bits


def hamming_distance(left: int, right: int) -> int:
    """Return the number of differing bits between two hashes."""
    return (left ^ right).bit_count()


@dataclass(frozen=True)
class DuplicateGroup:
    """One set of images whose perceptual hashes are near-identical."""

    original_ids: tuple[str, ...]
    hamming_distance: int


def find_duplicate_groups(
    manifests: dict[str, DatasetManifest],
    repo_root: Path,
    hamming_threshold: int = DEFAULT_HAMMING_THRESHOLD,
) -> list[DuplicateGroup]:
    """Group near-duplicate images across one or more manifests.

    `manifests` maps an arbitrary label (used only for error context) to a
    manifest; images are compared across all of them regardless of dataset,
    so this both finds accidental repeats within one split and cross-dataset
    overlap (e.g. a VisDrone-DET frame that also appears in VisDrone-MOT).
    """
    entries: list[tuple[str, ImageRecord, int]] = []
    for manifest in manifests.values():
        for sample in manifest.samples:
            image_path = repo_root / sample.relative_image_path
            entries.append((sample.original_id, sample, image_dhash(image_path)))

    groups: list[DuplicateGroup] = []
    assigned: set[int] = set()
    for index, (_original_id, _sample, digest) in enumerate(entries):
        if index in assigned:
            continue
        cluster_indices = [index]
        for other_index in range(index + 1, len(entries)):
            if other_index in assigned:
                continue
            distance = hamming_distance(digest, entries[other_index][2])
            if distance <= hamming_threshold:
                cluster_indices.append(other_index)
        if len(cluster_indices) > 1:
            assigned.update(cluster_indices)
            max_distance = max(
                hamming_distance(digest, entries[i][2]) for i in cluster_indices
            )
            groups.append(
                DuplicateGroup(
                    original_ids=tuple(entries[i][0] for i in cluster_indices),
                    hamming_distance=max_distance,
                )
            )
    return groups
