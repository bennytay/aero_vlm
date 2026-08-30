from pathlib import Path

from wam_drones.dataset.cli import build_parser


def test_cli_accepts_visdrone_det_download_and_convert(tmp_path: Path) -> None:
    parser = build_parser()

    download = parser.parse_args(["visdrone-det-download", "--split", "val"])
    assert download.split == "val"
    assert download.handler.__name__ == "run_visdrone_det_download"

    convert = parser.parse_args(
        ["visdrone-det-convert", "--source-dir", str(tmp_path)]
    )
    assert convert.split == "all"
    assert convert.source_dir == tmp_path


def test_cli_accepts_visdrone_mot_convert_with_required_args(tmp_path: Path) -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "visdrone-mot-convert",
            "--source-dir",
            str(tmp_path),
            "--split",
            "train",
            "--source-url",
            "manual-download://visdrone-mot-train",
        ]
    )

    assert args.source_dir == tmp_path
    assert args.dataset == "visdrone_mot"
    assert args.handler.__name__ == "run_visdrone_mot_convert"


def test_cli_accepts_uavdt_convert(tmp_path: Path) -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "uavdt-convert",
            "--source-dir",
            str(tmp_path),
            "--split",
            "train",
            "--source-url",
            "manual-download://uavdt-train",
        ]
    )

    assert args.handler.__name__ == "run_uavdt_convert"


def test_cli_accepts_dedup_validate_splits_and_report(tmp_path: Path) -> None:
    parser = build_parser()
    manifest_path = tmp_path / "manifest.json"

    dedup = parser.parse_args(["dedup", "--manifest", str(manifest_path)])
    assert dedup.manifest == [manifest_path]

    validate = parser.parse_args(
        ["validate-splits", "--manifest", str(manifest_path)]
    )
    assert validate.manifest == [manifest_path]

    report = parser.parse_args(["report", "--manifest", str(manifest_path)])
    assert report.sizes == "640,512,416,320"
