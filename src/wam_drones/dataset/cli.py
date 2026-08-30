"""Command-line entry points for Phase 2 dataset assembly."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from wam_drones.dataset.archive_config import load_archive_config, pin_split_sha256
from wam_drones.dataset.dedup import find_duplicate_groups
from wam_drones.dataset.manifest import DatasetManifest, DatasetSplit
from wam_drones.dataset.report import build_dataset_report
from wam_drones.dataset.splits import validate_split_integrity
from wam_drones.dataset.uavdt import convert_uavdt_split
from wam_drones.dataset.visdrone_det import (
    SPLIT_TO_ARCHIVE_NAME,
    convert_visdrone_det_split,
    download_visdrone_det_split,
)
from wam_drones.dataset.visdrone_mot import convert_visdrone_mot_split

ALL_DET_SPLITS: tuple[DatasetSplit, ...] = ("train", "val", "test")


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _requested_splits(split_arg: str) -> tuple[DatasetSplit, ...]:
    if split_arg == "all":
        return ALL_DET_SPLITS
    return (split_arg,)  # type: ignore[return-value]


def run_visdrone_det_download(args: argparse.Namespace) -> int:
    config = load_archive_config(args.config)
    for split in _requested_splits(args.split):
        extract_dir, sha256_hex = download_visdrone_det_split(
            split, args.dest_dir, config
        )
        if config.splits[split].sha256 is None:
            pin_split_sha256(args.config, split, sha256_hex)
            print(f"{split}: pinned sha256={sha256_hex}")
        print(f"{split}: {extract_dir}")
    return 0


def run_visdrone_det_convert(args: argparse.Namespace) -> int:
    config = load_archive_config(args.config)
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    for split in _requested_splits(args.split):
        source_dir = args.source_dir / "_extracted" / SPLIT_TO_ARCHIVE_NAME[split]
        manifest = convert_visdrone_det_split(
            source_dir,
            split,
            args.output_dir,
            args.repo_root,
            config.splits[split].url,
        )
        manifest_path = args.manifest_dir / f"visdrone_det_{split}_v1.json"
        manifest.write(manifest_path)
        print(f"{split}: {len(manifest.samples)} images -> {manifest_path}")
    return 0


def run_visdrone_mot_convert(args: argparse.Namespace) -> int:
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = convert_visdrone_mot_split(
        args.source_dir,
        args.split,
        args.output_dir,
        args.repo_root,
        args.source_url,
        args.dataset,
    )
    manifest_path = args.manifest_dir / f"{args.dataset}_{args.split}_v1.json"
    manifest.write(manifest_path)
    print(f"{args.split}: {len(manifest.samples)} frames -> {manifest_path}")
    return 0


def run_uavdt_convert(args: argparse.Namespace) -> int:
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = convert_uavdt_split(
        args.source_dir, args.split, args.output_dir, args.repo_root, args.source_url
    )
    manifest_path = args.manifest_dir / f"uavdt_{args.split}_v1.json"
    manifest.write(manifest_path)
    print(f"{args.split}: {len(manifest.samples)} frames -> {manifest_path}")
    return 0


def run_dedup(args: argparse.Namespace) -> int:
    manifests = {path.stem: DatasetManifest.read(path) for path in args.manifest}
    groups = find_duplicate_groups(manifests, args.repo_root, args.hamming_threshold)
    payload = {
        "hamming_threshold": args.hamming_threshold,
        "duplicate_group_count": len(groups),
        "groups": [
            {
                "original_ids": list(group.original_ids),
                "max_hamming_distance": group.hamming_distance,
            }
            for group in groups
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"{len(groups)} duplicate group(s) -> {args.output}")
    return 0


def run_validate_splits(args: argparse.Namespace) -> int:
    manifests = [DatasetManifest.read(path) for path in args.manifest]
    validate_split_integrity(manifests)
    print(f"OK: {len(manifests)} manifest(s) have disjoint splits")
    return 0


def run_report(args: argparse.Namespace) -> int:
    manifest = DatasetManifest.read(args.manifest)
    sizes = tuple(int(size) for size in args.sizes.split(","))
    report = build_dataset_report(manifest, args.repo_root, sizes)
    stem = f"{report.dataset}_{report.split}_report"
    json_path = args.output_dir / f"{stem}.json"
    markdown_path = args.output_dir / f"{stem}.md"
    report.write(json_path, markdown_path)
    print(f"{report.image_count} images, {report.box_count} boxes -> {json_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    root = repository_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=root)
    subparsers = parser.add_subparsers(dest="command", required=True)

    det_download = subparsers.add_parser(
        "visdrone-det-download", help="download and extract VisDrone-DET archives"
    )
    det_download.add_argument(
        "--split", choices=["train", "val", "test", "all"], default="all"
    )
    det_download.add_argument(
        "--config",
        type=Path,
        default=root / "configs" / "datasets" / "visdrone_det.yaml",
    )
    det_download.add_argument(
        "--dest-dir", type=Path, default=root / "data" / "raw" / "visdrone_det"
    )
    det_download.set_defaults(handler=run_visdrone_det_download)

    det_convert = subparsers.add_parser(
        "visdrone-det-convert", help="convert extracted VisDrone-DET to manifest+YOLO"
    )
    det_convert.add_argument(
        "--split", choices=["train", "val", "test", "all"], default="all"
    )
    det_convert.add_argument(
        "--config",
        type=Path,
        default=root / "configs" / "datasets" / "visdrone_det.yaml",
    )
    det_convert.add_argument(
        "--source-dir", type=Path, default=root / "data" / "raw" / "visdrone_det"
    )
    det_convert.add_argument(
        "--output-dir", type=Path, default=root / "data" / "raw" / "visdrone_det"
    )
    det_convert.add_argument(
        "--manifest-dir", type=Path, default=root / "data" / "manifests"
    )
    det_convert.set_defaults(handler=run_visdrone_det_convert)

    mot_convert = subparsers.add_parser(
        "visdrone-mot-convert",
        help="convert a locally extracted VisDrone-MOT/VID split to manifest+YOLO",
    )
    mot_convert.add_argument("--source-dir", type=Path, required=True)
    mot_convert.add_argument("--split", choices=["train", "val", "test"], required=True)
    mot_convert.add_argument("--source-url", required=True)
    mot_convert.add_argument(
        "--dataset", choices=["visdrone_mot", "visdrone_vid"], default="visdrone_mot"
    )
    mot_convert.add_argument(
        "--output-dir", type=Path, default=root / "data" / "raw" / "visdrone_mot"
    )
    mot_convert.add_argument(
        "--manifest-dir", type=Path, default=root / "data" / "manifests"
    )
    mot_convert.set_defaults(handler=run_visdrone_mot_convert)

    uavdt_convert = subparsers.add_parser(
        "uavdt-convert",
        help="convert a locally extracted UAVDT split to manifest+YOLO",
    )
    uavdt_convert.add_argument("--source-dir", type=Path, required=True)
    uavdt_convert.add_argument(
        "--split", choices=["train", "val", "test"], required=True
    )
    uavdt_convert.add_argument("--source-url", required=True)
    uavdt_convert.add_argument(
        "--output-dir", type=Path, default=root / "data" / "raw" / "uavdt"
    )
    uavdt_convert.add_argument(
        "--manifest-dir", type=Path, default=root / "data" / "manifests"
    )
    uavdt_convert.set_defaults(handler=run_uavdt_convert)

    dedup = subparsers.add_parser(
        "dedup", help="find near-duplicate images across one or more manifests"
    )
    dedup.add_argument("--manifest", type=Path, action="append", required=True)
    dedup.add_argument("--hamming-threshold", type=int, default=4)
    dedup.add_argument(
        "--output", type=Path, default=root / "runs" / "dataset" / "dedup_report.json"
    )
    dedup.set_defaults(handler=run_dedup)

    validate_splits = subparsers.add_parser(
        "validate-splits", help="reject images or sequences spanning multiple splits"
    )
    validate_splits.add_argument(
        "--manifest", type=Path, action="append", required=True
    )
    validate_splits.set_defaults(handler=run_validate_splits)

    report = subparsers.add_parser(
        "report", help="class counts, area histogram, and letterbox-shrink report"
    )
    report.add_argument("--manifest", type=Path, required=True)
    report.add_argument("--sizes", default="640,512,416,320")
    report.add_argument(
        "--output-dir", type=Path, default=root / "runs" / "dataset" / "reports"
    )
    report.set_defaults(handler=run_report)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
