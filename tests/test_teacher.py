from pathlib import Path

from wam_drones.contracts import TargetObservation
from wam_drones.teacher.model import NO_TARGET, TeacherResult, run_target


class FakeTeacher:
    def __init__(self, predicted_label: str) -> None:
        self.predicted_label = predicted_label

    def classify(self, image_path: Path, template_name: str) -> TeacherResult:
        del image_path, template_name
        return TeacherResult(
            predicted_label=self.predicted_label,
            confidence=0.8,
            probabilities={self.predicted_label: 0.8},
            similarities={self.predicted_label: 0.3},
            image_embedding=(0.1, 0.2),
            latency_ms=1.5,
        )


def test_run_target_returns_visible_observation_contract() -> None:
    observation = run_target(
        Path("fixture.jpg"), FakeTeacher("bicycle"), timestamp_us=123
    )

    assert isinstance(observation, TargetObservation)
    assert observation.visible
    assert observation.target_label == "bicycle"
    assert observation.target_id == 7
    assert observation.bbox_xyxy_normalized == (0.0, 0.0, 1.0, 1.0)


def test_run_target_returns_no_target_observation_contract() -> None:
    observation = run_target(
        Path("fixture.jpg"), FakeTeacher(NO_TARGET), timestamp_us=123
    )

    assert isinstance(observation, TargetObservation)
    assert not observation.visible
    assert observation.target_label is None
    assert observation.bbox_xyxy_normalized is None
