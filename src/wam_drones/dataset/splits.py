"""Cross-split integrity checks: no leaked images, no split sequences."""

from __future__ import annotations

from collections.abc import Sequence

from wam_drones.dataset.manifest import DatasetManifest


def validate_split_integrity(manifests: Sequence[DatasetManifest]) -> None:
    """Raise if any image or sequence spans more than one dataset split.

    A `sha256` (or `sequence_id`) appearing under two different splits means
    the same content could leak between train and validation/test, silently
    inflating measured accuracy.
    """
    sha256_to_split: dict[str, str] = {}
    sequence_to_split: dict[str, str] = {}
    for manifest in manifests:
        for sample in manifest.samples:
            split_key = f"{manifest.dataset}:{manifest.split}"
            previous_split = sha256_to_split.get(sample.sha256)
            if previous_split is not None and previous_split != split_key:
                raise ValueError(
                    f"image {sample.sha256} appears in both "
                    f"{previous_split} and {split_key}"
                )
            sha256_to_split[sample.sha256] = split_key

            if sample.sequence_id is None:
                continue
            sequence_key = f"{manifest.dataset}:{sample.sequence_id}"
            previous_sequence_split = sequence_to_split.get(sequence_key)
            if (
                previous_sequence_split is not None
                and previous_sequence_split != split_key
            ):
                raise ValueError(
                    f"sequence {sample.sequence_id} appears in both "
                    f"{previous_sequence_split} and {split_key}"
                )
            sequence_to_split[sequence_key] = split_key
