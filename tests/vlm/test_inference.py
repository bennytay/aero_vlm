from pathlib import Path

import pytest

from wam_drones.vlm.cli import build_parser
from wam_drones.vlm.inference import infer_image
from wam_drones.vlm.model_config import load_model_config


class FakeBackend:
    model_revision = "test/model@abc123"
    preprocessing_dimensions = (512, 384)

    def __init__(self, output: str) -> None:
        self.output = output

    def generate(self, rgb: object, question: str) -> str:
        assert question == "How many cars are visible?"
        return self.output


def _image(tmp_path: Path) -> Path:
    image_module = pytest.importorskip("PIL.Image")
    path = tmp_path / "frame.png"
    image_module.new("RGB", (20, 10), "white").save(path)
    return path


def test_inference_preserves_invalid_generation_in_audit_record(tmp_path: Path) -> None:
    record, _ = infer_image(
        _image(tmp_path), FakeBackend("not valid json"), "How many cars are visible?"
    )
    assert record.response is None
    assert record.parse_errors
    assert record.preprocessing.model_width_px == 512


def test_inference_validates_generation_without_repair(tmp_path: Path) -> None:
    record, _ = infer_image(
        _image(tmp_path),
        FakeBackend(
            '{"schema_version":"vlm_response_v1","type":"answer",'
            '"status":"ok","answer":{"kind":"count","value":2}}'
        ),
        "How many cars are visible?",
    )
    assert record.response is not None
    assert record.response.type == "answer"
    assert not record.parse_errors


def test_model_configs_are_deterministic_and_include_three_candidates() -> None:
    root = Path(__file__).parents[2]
    configs = [
        load_model_config(path)
        for path in sorted((root / "configs/vlm/models").glob("*.yaml"))
    ]
    assert {config.name for config in configs} == {
        "qwen3-vl-4b",
        "qwen3.5-4b",
        "miril-dronevlm-2b-2",
    }
    assert all(not config.do_sample and config.temperature == 0 for config in configs)


def test_infer_cli_accepts_a_single_image_question() -> None:
    args = build_parser().parse_args(
        ["infer", "frame.jpg", "--question", "What is visible?"]
    )
    assert args.command == "infer"
    assert args.question == "What is visible?"
