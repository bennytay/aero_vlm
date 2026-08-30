from pathlib import Path

import pytest

from wam_drones.detection.inference import (
    ModelPrediction,
    RawDetection,
    detect_image,
    iter_image_folder,
    prediction_to_contract,
)
from wam_drones.detection.model_config import (
    file_sha256,
    load_model_config,
    verify_checkpoint,
)
from wam_drones.detection.vocabulary import DetectionLabel


class FakeBackend:
    model_name = "fake-detector"

    def predict(self, image: object, *, track: bool = False) -> ModelPrediction:
        del image, track
        return ModelPrediction(
            image_width_px=200,
            image_height_px=100,
            detections=(
                RawDetection("car", 0.9, (20.0, 10.0, 100.0, 60.0)),
                RawDetection("dog", 0.8, (0.0, 0.0, 20.0, 20.0)),
                RawDetection("truck", 0.7, (-10.0, 5.0, 220.0, 95.0)),
                RawDetection("person", 0.6, (10.0, 20.0, 10.0, 40.0)),
            ),
            annotated_bgr="rendered",
        )


MAPPING = {
    "person": DetectionLabel.PERSON,
    "car": DetectionLabel.CAR,
    "truck": DetectionLabel.TRUCK,
}


def test_prediction_converts_supported_classes_and_clamps_boxes() -> None:
    prediction = FakeBackend().predict("unused")
    frame = prediction_to_contract(
        prediction,
        frame_id=3,
        captured_at_monotonic_ns=10,
        inference_started_at_monotonic_ns=11,
        produced_at_monotonic_ns=12,
        model_name="fake-detector",
        class_mapping=MAPPING,
    )

    assert [detection.label for detection in frame.detections] == ["car", "truck"]
    assert frame.detections[0].bbox_norm_xyxy == (0.1, 0.1, 0.5, 0.6)
    assert frame.detections[1].bbox_norm_xyxy == (0.0, 0.05, 1.0, 0.95)
    assert frame.detections[0].label_id == 4


def test_detect_image_preserves_explicit_capture_time() -> None:
    frame, rendered = detect_image(
        "unused", FakeBackend(), MAPPING, frame_id=9, captured_at_monotonic_ns=100
    )

    assert frame.frame_id == 9
    assert frame.captured_at_monotonic_ns == 100
    assert frame.inference_started_at_monotonic_ns >= 100
    assert rendered == "rendered"

    zero_time, _ = detect_image(
        "unused", FakeBackend(), MAPPING, captured_at_monotonic_ns=0
    )
    assert zero_time.captured_at_monotonic_ns == 0


def test_image_folder_is_stably_ordered(tmp_path: Path) -> None:
    for name in ("b.JPG", "ignore.txt", "a.png"):
        (tmp_path / name).touch()

    results = list(iter_image_folder(tmp_path, FakeBackend(), MAPPING))

    assert [path.name for path, _, _ in results] == ["a.png", "b.JPG"]
    assert [frame.frame_id for _, frame, _ in results] == [0, 1]


def test_model_config_and_checkpoint_hash_are_pinned(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    config = load_model_config(root / "configs" / "models" / "yolo26n_coco.yaml")
    assert config.package_version == "8.4.135"
    assert config.checkpoint_sha256 == (
        "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
    )
    assert config.class_mapping["motorcycle"] is DetectionLabel.MOTOR

    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"known bytes")
    digest = file_sha256(checkpoint)
    verify_checkpoint(checkpoint, digest)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_checkpoint(checkpoint, "0" * 64)
