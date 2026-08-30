from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from wam_drones.dataset.dedup import (
    find_duplicate_groups,
    hamming_distance,
    image_dhash,
)
from wam_drones.dataset.manifest import DatasetManifest, ImageRecord

VALID_SHA = "sha256:" + "b" * 64


def make_manifest(dataset: str, split: str, image_paths: list[Path]) -> DatasetManifest:
    samples = tuple(
        ImageRecord(
            dataset=dataset,  # type: ignore[arg-type]
            split=split,  # type: ignore[arg-type]
            source_url="https://example.invalid",
            original_id=path.stem,
            relative_image_path=path.name,
            sha256=VALID_SHA,
            width_px=64,
            height_px=64,
            annotation_conversion_version="v1",
            box_count=0,
            ignored_region_count=0,
        )
        for path in image_paths
    )
    return DatasetManifest(
        dataset=dataset,  # type: ignore[arg-type]
        split=split,  # type: ignore[arg-type]
        created_utc=datetime(2026, 8, 31, tzinfo=UTC),
        samples=samples,
    )


def horizontal_gradient() -> Image.Image:
    image = Image.new("RGB", (64, 64))
    for x in range(64):
        for y in range(64):
            image.putpixel((x, y), (x * 4 % 256, x * 4 % 256, x * 4 % 256))
    return image


def mirrored_gradient() -> Image.Image:
    """A brightness-decreasing gradient: dHash flips every bit vs. increasing."""
    image = Image.new("RGB", (64, 64))
    for x in range(64):
        for y in range(64):
            value = 252 - x * 4 % 256
            image.putpixel((x, y), (value, value, value))
    return image


def test_hamming_distance_of_identical_hashes_is_zero() -> None:
    assert hamming_distance(0b1010, 0b1010) == 0
    assert hamming_distance(0b1010, 0b0010) == 1


def test_image_dhash_flags_re_saved_copy_as_near_duplicate(tmp_path: Path) -> None:
    original = tmp_path / "original.jpg"
    resaved = tmp_path / "resaved.jpg"
    horizontal_gradient().save(original, quality=95)
    Image.open(original).save(resaved, quality=60)

    assert hamming_distance(image_dhash(original), image_dhash(resaved)) <= 4


def test_find_duplicate_groups_detects_duplicate_across_manifests(
    tmp_path: Path,
) -> None:
    shared = horizontal_gradient()
    det_path = tmp_path / "det.jpg"
    mot_path = tmp_path / "mot.jpg"
    distinct_path = tmp_path / "distinct.jpg"
    shared.save(det_path)
    shared.save(mot_path)
    mirrored_gradient().save(distinct_path)

    manifests = {
        "det": make_manifest("visdrone_det", "train", [det_path, distinct_path]),
        "mot": make_manifest("visdrone_mot", "train", [mot_path]),
    }

    groups = find_duplicate_groups(manifests, tmp_path)

    assert len(groups) == 1
    assert set(groups[0].original_ids) == {"det", "mot"}


def test_find_duplicate_groups_returns_empty_for_distinct_images(
    tmp_path: Path,
) -> None:
    path_a = tmp_path / "a.jpg"
    path_b = tmp_path / "b.jpg"
    horizontal_gradient().save(path_a)
    mirrored_gradient().save(path_b)

    manifests = {"det": make_manifest("visdrone_det", "train", [path_a, path_b])}

    assert find_duplicate_groups(manifests, tmp_path) == []
