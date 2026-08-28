"""Pure rejection-threshold decision and sweep functions."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, cast

from wam_drones.scope import SUPPORTED_LABELS
from wam_drones.vocabulary import TARGET_IDS_BY_LABEL

# Updated only from exp_20260828_teacher_threshold evidence.
DEFAULT_SIMILARITY_THRESHOLD: Final = 0.17


@dataclass(frozen=True)
class ThresholdDecision:
    """Result of applying the flight rejection rule to ten cosine scores."""

    visible: bool
    predicted_label: str | None
    best_label: str
    best_score: float


@dataclass(frozen=True)
class ScoredSample:
    """Ground truth and ten frozen text similarities for one image."""

    expected_label: str | None
    similarities: Mapping[str, float]


def apply_reject_threshold(
    similarities: Mapping[str, float], threshold: float
) -> ThresholdDecision:
    """Apply argmax over ten names, then reject when the best cosine is below T."""
    missing = set(TARGET_IDS_BY_LABEL).difference(similarities)
    if missing:
        raise ValueError(f"missing target similarities: {sorted(missing)}")
    best_label = max(TARGET_IDS_BY_LABEL, key=similarities.__getitem__)
    best_score = float(similarities[best_label])
    visible = best_score >= threshold
    return ThresholdDecision(
        visible=visible,
        predicted_label=best_label if visible else None,
        best_label=best_label,
        best_score=best_score,
    )


def threshold_metrics(
    samples: Sequence[ScoredSample],
    threshold: float,
    supported_labels: Sequence[str] = SUPPORTED_LABELS,
) -> dict[str, object]:
    """Measure target accuracy, supported recall, and no-target rejection."""
    target_samples = [sample for sample in samples if sample.expected_label is not None]
    no_target_samples = [sample for sample in samples if sample.expected_label is None]
    target_correct = 0
    false_rejects = 0
    supported_total = {label: 0 for label in supported_labels}
    supported_correct = {label: 0 for label in supported_labels}
    for sample in target_samples:
        decision = apply_reject_threshold(sample.similarities, threshold)
        if not decision.visible:
            false_rejects += 1
        if decision.predicted_label == sample.expected_label:
            target_correct += 1
        if sample.expected_label in supported_total:
            supported_total[sample.expected_label] += 1
            if decision.predicted_label == sample.expected_label:
                supported_correct[sample.expected_label] += 1
    no_target_false_positives = sum(
        apply_reject_threshold(sample.similarities, threshold).visible
        for sample in no_target_samples
    )
    false_positive_rate = (
        no_target_false_positives / len(no_target_samples) if no_target_samples else 0.0
    )
    per_class_recall = {
        label: (
            supported_correct[label] / supported_total[label]
            if supported_total[label]
            else None
        )
        for label in supported_labels
    }
    recalls = [value for value in per_class_recall.values() if value is not None]
    return {
        "threshold": threshold,
        "target_top1": target_correct / len(target_samples) if target_samples else 0.0,
        "supported_per_class_recall": per_class_recall,
        "supported_macro_recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "no_target_false_positive_rate": false_positive_rate,
        "no_target_tnr": 1.0 - false_positive_rate,
        "rejected_target_images": false_rejects,
        "target_images": len(target_samples),
        "no_target_images": len(no_target_samples),
    }


def sweep_thresholds(
    samples: Sequence[ScoredSample], thresholds: Sequence[float]
) -> list[dict[str, object]]:
    """Evaluate an ordered deterministic threshold grid."""
    return [threshold_metrics(samples, threshold) for threshold in thresholds]


def choose_threshold(
    curve: Sequence[Mapping[str, object]], target_false_positive_rate: float = 0.20
) -> Mapping[str, object]:
    """Choose the highest-recall point meeting the no-target FP target."""
    if not curve:
        raise ValueError("threshold curve is empty")

    def number(point: Mapping[str, object], key: str) -> float:
        return float(cast(float | int, point[key]))

    feasible = [
        point
        for point in curve
        if number(point, "no_target_false_positive_rate") <= target_false_positive_rate
    ]
    candidates = feasible or list(curve)
    return max(
        candidates,
        key=lambda point: (
            number(point, "supported_macro_recall"),
            -number(point, "no_target_false_positive_rate"),
            -number(point, "threshold"),
        ),
    )
