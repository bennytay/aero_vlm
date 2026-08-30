from pathlib import Path

import pytest
from pydantic import ValidationError

from wam_drones.detection import DetectionLabel, load_detection_vocabulary
from wam_drones.detection.vocabulary import DetectionVocabulary


def test_detection_vocabulary_loads_all_native_classes() -> None:
    repo_root = Path(__file__).parents[1]
    vocabulary = load_detection_vocabulary(
        repo_root / "configs" / "vocabulary_detection_v1.yaml"
    )

    assert vocabulary.version == "v1"
    assert tuple(label.id for label in vocabulary.labels) == tuple(range(1, 11))
    assert tuple(label.name for label in vocabulary.labels) == tuple(DetectionLabel)


def test_detection_vocabulary_rejects_reordered_or_duplicate_ids() -> None:
    labels: list[dict[str, int | str]] = [
        {"id": index, "name": label.value}
        for index, label in enumerate(DetectionLabel, start=1)
    ]
    labels[1]["id"] = 1
    payload = {
        "version": "v1",
        "source_taxonomy": "VisDrone",
        "labels": labels,
    }

    with pytest.raises(ValidationError, match="exactly match"):
        DetectionVocabulary.model_validate(payload)
