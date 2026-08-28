"""Assemble the licensed held-out v0 image subset without scraping."""

import argparse
import csv
import json
import shutil
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from wam_drones.dataset import DatasetManifest, DatasetSample
from wam_drones.vocabulary import TARGET_IDS_BY_LABEL

OPEN_IMAGES_FILES = {
    "boxes": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
    "labels": "https://storage.googleapis.com/openimages/v7/oidv7-val-annotations-human-imagelabels.csv",
    "images": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
}
PPE_VALID_ZIP = (
    "https://huggingface.co/datasets/keremberke/"
    "construction-safety-object-detection/resolve/main/data/valid.zip"
)
OPEN_IMAGES_MIDS = {
    "red backpack": ("/m/01940j",),
    "water bottle": ("/m/04dr76w",),
    "black vehicle": ("/m/0k4j", "/m/07r04", "/m/0h2r6"),
    "solar panel": ("/m/0fnrbc",),
    "wheelie bin": ("/m/0bjyj5",),
    "bicycle": ("/m/0199g",),
    "cardboard box": ("/m/025dyy",),
    "sports ball": ("/m/018xm",),
    "orange bucket": ("/m/01zfyk",),
}
IMAGE_LEVEL_ONLY = frozenset({"solar panel", "orange bucket"})


def _download(url: str, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(f"{destination.suffix}.part")
    request = urllib.request.Request(url, headers={"User-Agent": "wam-drones/0.1"})
    try:
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            partial.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)
        partial.replace(destination)
    except OSError:
        partial.unlink(missing_ok=True)
        raise


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _bbox(row: dict[str, str]) -> tuple[float, float, float, float]:
    return (
        float(row["XMin"]),
        float(row["YMin"]),
        float(row["XMax"]),
        float(row["YMax"]),
    )


def _read_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        yield from csv.DictReader(source)


def _image_metadata(path: Path) -> dict[str, dict[str, str]]:
    return {row["ImageID"]: row for row in _read_rows(path)}


def _positive_image_labels(path: Path) -> dict[str, set[str]]:
    positives: dict[str, set[str]] = defaultdict(set)
    mapped_mids = {mid for mids in OPEN_IMAGES_MIDS.values() for mid in mids}
    for row in _read_rows(path):
        if row["Confidence"] == "1" and row["LabelName"] in mapped_mids:
            positives[row["ImageID"]].add(row["LabelName"])
    return positives


def _box_candidates(
    path: Path,
) -> tuple[dict[str, dict[str, list[tuple[float, float, float, float]]]], set[str]]:
    labels_by_mid = {
        mid: target for target, mids in OPEN_IMAGES_MIDS.items() for mid in mids
    }
    candidates: dict[str, dict[str, list[tuple[float, float, float, float]]]] = {
        label: defaultdict(list) for label in OPEN_IMAGES_MIDS
    }
    all_box_images: set[str] = set()
    for row in _read_rows(path):
        image_id = row["ImageID"]
        all_box_images.add(image_id)
        target = labels_by_mid.get(row["LabelName"])
        if target is not None:
            candidates[target][image_id].append(_bbox(row))
    return candidates, all_box_images


def _union_boxes(
    boxes: Sequence[tuple[float, float, float, float]],
) -> tuple[float, float, float, float] | None:
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _download_open_image(
    metadata: dict[str, str], image_id: str, raw_root: Path
) -> tuple[Path, str]:
    destination = raw_root / "open_images" / "val" / f"{image_id}.jpg"
    urls = [metadata.get("Thumbnail300KURL", ""), metadata["OriginalURL"]]
    for url in dict.fromkeys(url for url in urls if url):
        try:
            _download(url, destination)
        except urllib.error.URLError:
            continue
        return destination, url
    raise urllib.error.URLError(f"no live URL for Open Images ID {image_id}")


def _download_open_images(
    image_ids: Sequence[str],
    metadata: dict[str, dict[str, str]],
    raw_root: Path,
) -> list[tuple[Path, str] | None]:
    def download_one(image_id: str) -> tuple[Path, str] | None:
        try:
            return _download_open_image(metadata[image_id], image_id, raw_root)
        except (KeyError, OSError):
            return None

    with ThreadPoolExecutor(max_workers=16) as executor:
        return list(executor.map(download_one, image_ids))


def _relative(path: Path, repo_root: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def open_images_samples(
    repo_root: Path, *, max_per_label: int, max_no_target: int
) -> list[DatasetSample]:
    download_root = repo_root / "data" / "downloads" / "open_images"
    for name, url in OPEN_IMAGES_FILES.items():
        _download(url, download_root / f"{name}.csv")
    metadata = _image_metadata(download_root / "images.csv")
    positives = _positive_image_labels(download_root / "labels.csv")
    box_candidates, all_box_images = _box_candidates(download_root / "boxes.csv")
    all_mapped_mids = {mid for mids in OPEN_IMAGES_MIDS.values() for mid in mids}
    labels_by_mid = {
        mid: target for target, mids in OPEN_IMAGES_MIDS.items() for mid in mids
    }
    targets_by_image: dict[str, set[str]] = defaultdict(set)
    for target, candidates in box_candidates.items():
        for image_id in candidates:
            targets_by_image[image_id].add(target)
    for image_id, positive_mids in positives.items():
        targets_by_image[image_id].update(
            labels_by_mid[mid] for mid in positive_mids
        )
    samples: list[DatasetSample] = []
    used_images: set[str] = set()
    raw_root = repo_root / "data" / "raw" / "v0"

    for target_label, mids in OPEN_IMAGES_MIDS.items():
        if target_label in IMAGE_LEVEL_ONLY:
            image_ids = sorted(
                image_id
                for image_id, labels in positives.items()
                if labels.intersection(mids)
                and targets_by_image[image_id] == {target_label}
            )
        else:
            image_ids = sorted(
                image_id
                for image_id in box_candidates[target_label]
                if targets_by_image[image_id] == {target_label}
            )
        selected_count = 0
        available_ids = [
            image_id for image_id in image_ids if image_id not in used_images
        ]
        for offset in range(0, len(available_ids), 32):
            batch_ids = available_ids[offset : offset + 32]
            downloaded = _download_open_images(batch_ids, metadata, raw_root)
            for image_id, download_result in zip(batch_ids, downloaded, strict=True):
                if download_result is None:
                    continue
                image_path, source_url = download_result
                samples.append(
                    DatasetSample(
                        image_path=_relative(image_path, repo_root),
                        image_sha256=_sha256(image_path),
                        split="val",
                        source="open_images_v7",
                        source_id=image_id,
                        session_id=f"open_images:{image_id}",
                        source_url=source_url,
                        licence=metadata[image_id]["License"],
                        no_target=False,
                        target_label=target_label,
                        target_id=TARGET_IDS_BY_LABEL[target_label],
                        bbox_xyxy_normalized=_union_boxes(
                            box_candidates[target_label].get(image_id, [])
                        ),
                        colour_verified=False,
                    )
                )
                used_images.add(image_id)
                selected_count += 1
                if selected_count >= max_per_label:
                    break
            if selected_count >= max_per_label:
                break

    positive_target_images = {
        image_id
        for image_id, labels in positives.items()
        if labels.intersection(all_mapped_mids)
    }
    positive_target_images.update(
        image_id
        for candidates in box_candidates.values()
        for image_id in candidates
    )
    no_target_ids = sorted(all_box_images - positive_target_images - used_images)
    no_target_count = 0
    for offset in range(0, len(no_target_ids), 32):
        batch_ids = no_target_ids[offset : offset + 32]
        downloaded = _download_open_images(batch_ids, metadata, raw_root)
        for image_id, download_result in zip(batch_ids, downloaded, strict=True):
            if download_result is None:
                continue
            image_path, source_url = download_result
            samples.append(
                DatasetSample(
                    image_path=_relative(image_path, repo_root),
                    image_sha256=_sha256(image_path),
                    split="val",
                    source="open_images_v7",
                    source_id=image_id,
                    session_id=f"open_images:{image_id}",
                    source_url=source_url,
                    licence=metadata[image_id]["License"],
                    no_target=True,
                )
            )
            no_target_count += 1
            if no_target_count >= max_no_target:
                break
        if no_target_count >= max_no_target:
            break
    return samples


def _load_coco(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def ppe_samples(repo_root: Path, *, max_per_label: int) -> list[DatasetSample]:
    download_path = repo_root / "data" / "downloads" / "ppe" / "valid.zip"
    extract_root = repo_root / "data" / "downloads" / "ppe" / "valid"
    _download(PPE_VALID_ZIP, download_path)
    if not extract_root.exists():
        with zipfile.ZipFile(download_path) as archive:
            archive.extractall(extract_root)
    annotations_path = next(extract_root.rglob("*.json"))
    coco = _load_coco(annotations_path)
    categories = {
        int(category["id"]): str(category["name"])
        for category in cast(list[dict[str, Any]], coco["categories"])
    }
    images = {
        int(image["id"]): image
        for image in cast(list[dict[str, Any]], coco["images"])
    }
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in cast(list[dict[str, Any]], coco["annotations"]):
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    samples: list[DatasetSample] = []
    raw_root = repo_root / "data" / "raw" / "v0" / "ppe" / "val"
    raw_root.mkdir(parents=True, exist_ok=True)
    for image_id in sorted(images):
        image = images[image_id]
        safety_boxes = [
            cast(list[float], annotation["bbox"])
            for annotation in annotations_by_image[image_id]
            if categories[int(annotation["category_id"])] == "safety vest"
        ]
        if not safety_boxes:
            continue
        source_path = next(extract_root.rglob(str(image["file_name"])))
        destination = raw_root / source_path.name
        if not destination.exists():
            shutil.copy2(source_path, destination)
        width, height = float(image["width"]), float(image["height"])
        normalised = [
            (x / width, y / height, (x + w) / width, (y + h) / height)
            for x, y, w, h in safety_boxes
        ]
        samples.append(
            DatasetSample(
                image_path=_relative(destination, repo_root),
                image_sha256=_sha256(destination),
                split="val",
                source="construction_safety_cc_by_4",
                source_id=str(image_id),
                session_id=f"construction_safety:{Path(source_path.name).stem}",
                source_url=(
                    "https://huggingface.co/datasets/keremberke/"
                    "construction-safety-object-detection"
                ),
                licence="CC BY 4.0",
                no_target=False,
                target_label="person in high-vis",
                target_id=TARGET_IDS_BY_LABEL["person in high-vis"],
                bbox_xyxy_normalized=_union_boxes(normalised),
                colour_verified=True,
            )
        )
        if len(samples) >= max_per_label:
            break
    return samples


def assemble(
    repo_root: Path, *, max_per_label: int = 200, max_no_target: int = 300
) -> Path:
    samples = open_images_samples(
        repo_root,
        max_per_label=max_per_label,
        max_no_target=max_no_target,
    )
    samples.extend(ppe_samples(repo_root, max_per_label=max_per_label))
    samples.sort(key=lambda sample: (sample.target_id or 255, sample.session_id))
    manifest = DatasetManifest(
        created_utc=datetime.now(UTC),
        samples=tuple(samples),
    )
    output = repo_root / "data" / "manifests" / "dataset_v0.json"
    manifest.write(output)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--max-per-label", type=int, default=200)
    parser.add_argument("--max-no-target", type=int, default=300)
    args = parser.parse_args(argv)
    path = assemble(
        args.repo_root.resolve(),
        max_per_label=args.max_per_label,
        max_no_target=args.max_no_target,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
