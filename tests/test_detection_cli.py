from pathlib import Path

from wam_drones.detection.cli import build_parser


def test_cli_accepts_folder_video_export_and_parity_commands(tmp_path: Path) -> None:
    parser = build_parser()
    run = parser.parse_args(["run", str(tmp_path), "--no-track"])
    assert run.source == tmp_path
    assert run.track is False

    export = parser.parse_args(["export", "--output-dir", str(tmp_path)])
    assert export.output_dir == tmp_path

    parity = parser.parse_args(
        ["parity", "image.jpg", "--onnx", str(tmp_path / "model.onnx")]
    )
    assert parity.onnx == tmp_path / "model.onnx"

    prepare = parser.parse_args(["phase3-prepare", "--output-dir", str(tmp_path)])
    assert prepare.output_dir == tmp_path

    evaluate = parser.parse_args(
        [
            "phase3-evaluate",
            "--checkpoint",
            "model.pt",
            "--model-kind",
            "native",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert evaluate.model_kind == "native"

    demo = parser.parse_args(
        [
            "demo-video",
            "my-footage.mov",
            "--output",
            str(tmp_path / "demo.mp4"),
            "--detector-cadence",
            "2",
        ]
    )
    assert demo.source == Path("my-footage.mov")
    assert demo.output == tmp_path / "demo.mp4"
    assert demo.detector_cadence == 2
