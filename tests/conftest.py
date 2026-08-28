from collections.abc import Callable

import pytest

from wam_drones.contracts import TargetObservation


@pytest.fixture
def target_observation_factory() -> Callable[[], TargetObservation]:
    """Build a valid, visible observation for contract tests and future producers."""

    def build() -> TargetObservation:
        return TargetObservation(
            timestamp_us=1_000_000,
            visible=True,
            confidence=0.91,
            center_x_normalized=0.5,
            center_y_normalized=0.45,
            bbox_xyxy_normalized=(0.25, 0.2, 0.75, 0.7),
            target_label="red backpack",
            target_id=1,
            stale=False,
        )

    return build
