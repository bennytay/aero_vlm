from pathlib import Path
from typing import Any, cast

import yaml

from wam_drones.threshold import (
    DEFAULT_SIMILARITY_THRESHOLD,
    ScoredSample,
    apply_reject_threshold,
    choose_threshold,
    sweep_thresholds,
)
from wam_drones.vocabulary import TARGET_IDS_BY_LABEL


def scores(best_label: str, best_score: float) -> dict[str, float]:
    values = dict.fromkeys(TARGET_IDS_BY_LABEL, 0.1)
    values[best_label] = best_score
    return values


def test_threshold_rule_is_pure_and_rejects_below_t() -> None:
    similarities = scores("bicycle", 0.24)

    first = apply_reject_threshold(similarities, 0.25)
    second = apply_reject_threshold(similarities, 0.25)

    assert first == second
    assert not first.visible
    assert first.predicted_label is None
    assert first.best_label == "bicycle"
    assert first.best_score == 0.24


def test_frozen_threshold_matches_config() -> None:
    document = cast(
        dict[str, Any],
        yaml.safe_load(
            (
                Path(__file__).parents[1] / "configs" / "teacher_threshold_v0.yaml"
            ).read_text(encoding="utf-8")
        ),
    )

    assert document["similarity_threshold"] == DEFAULT_SIMILARITY_THRESHOLD


def test_threshold_rule_accepts_at_t() -> None:
    decision = apply_reject_threshold(scores("water bottle", 0.25), 0.25)

    assert decision.visible
    assert decision.predicted_label == "water bottle"


def test_sweep_and_choice_meet_false_positive_target() -> None:
    samples = [
        ScoredSample("bicycle", scores("bicycle", 0.4)),
        ScoredSample("water bottle", scores("water bottle", 0.3)),
        ScoredSample(None, scores("sports ball", 0.28)),
        ScoredSample(None, scores("sports ball", 0.18)),
    ]
    curve = sweep_thresholds(samples, [0.1, 0.2, 0.3])

    chosen = choose_threshold(curve, target_false_positive_rate=0.5)

    assert chosen["threshold"] == 0.3
    assert chosen["no_target_false_positive_rate"] == 0.0
    assert chosen["rejected_target_images"] == 0
