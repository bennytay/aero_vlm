"""Evaluate two frozen prompt templates on the scene-separated v0 validation set."""

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from statistics import median
from typing import Any

from wam_drones.dataset import DatasetManifest, DatasetSample
from wam_drones.teacher.model import (
    NO_TARGET,
    PROMPT_TEMPLATES,
    MobileClipTeacher,
    TeacherBackend,
)
from wam_drones.vocabulary import TARGET_IDS_BY_LABEL


def percentile(values: Sequence[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile_value)
    return ordered[index]


def expected_label(sample: DatasetSample) -> str:
    return NO_TARGET if sample.no_target else str(sample.target_label)


def evaluate(
    repo_root: Path,
    manifest: DatasetManifest,
    output_dir: Path,
    backend: TeacherBackend,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.csv"
    embeddings_path = output_dir / "embeddings.jsonl.gz"
    fieldnames = [
        "template",
        "image_path",
        "source_id",
        "expected_label",
        "predicted_label",
        "confidence",
        "top3",
        "latency_ms",
        "correct_top1",
        "correct_top3",
    ]
    rows: list[dict[str, str | float | int]] = []
    embeddings: list[dict[str, Any]] = []
    results_by_template: dict[str, list[tuple[DatasetSample, Any]]] = defaultdict(list)
    for template_name in PROMPT_TEMPLATES:
        for sample in manifest.samples:
            image_path = repo_root / sample.image_path
            result = backend.classify(image_path, template_name)
            ranked = sorted(
                result.probabilities,
                key=result.probabilities.__getitem__,
                reverse=True,
            )
            truth = expected_label(sample)
            rows.append(
                {
                    "template": template_name,
                    "image_path": sample.image_path,
                    "source_id": sample.source_id,
                    "expected_label": truth,
                    "predicted_label": result.predicted_label,
                    "confidence": result.confidence,
                    "top3": "|".join(ranked[:3]),
                    "latency_ms": result.latency_ms,
                    "correct_top1": int(result.predicted_label == truth),
                    "correct_top3": int(truth in ranked[:3]),
                }
            )
            embeddings.append(
                {
                    "template": template_name,
                    "source_id": sample.source_id,
                    "embedding": result.image_embedding,
                }
            )
            results_by_template[template_name].append((sample, result))
    with predictions_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with gzip.open(embeddings_path, "wt", encoding="utf-8") as output:
        for row in embeddings:
            output.write(json.dumps(row, separators=(",", ":")) + "\n")

    metrics: dict[str, Any] = {
        "chance_top1": 1 / 11,
        "n_total": len(manifest.samples),
        "n_per_class": dict(
            sorted(Counter(expected_label(s) for s in manifest.samples).items())
        ),
        "templates": {},
    }
    labels = list(TARGET_IDS_BY_LABEL)
    gallery_lines = ["# Error gallery", "", "Paths refer to ignored local images.", ""]
    for template_name, results in results_by_template.items():
        latencies = [result.latency_ms for _, result in results]
        top1 = [
            result.predicted_label == expected_label(sample)
            for sample, result in results
        ]
        top3 = [
            expected_label(sample)
            in sorted(
                result.probabilities,
                key=result.probabilities.__getitem__,
                reverse=True,
            )[:3]
            for sample, result in results
        ]
        no_target_results = [result for sample, result in results if sample.no_target]
        target_results = [
            (sample, result) for sample, result in results if not sample.no_target
        ]
        confusion = {
            truth: {prediction: 0 for prediction in labels} for truth in labels
        }
        predicted_no_target = Counter[str]()
        per_class_correct: Counter[str] = Counter()
        per_class_total: Counter[str] = Counter()
        for sample, result in target_results:
            truth = str(sample.target_label)
            per_class_total[truth] += 1
            if result.predicted_label == truth:
                per_class_correct[truth] += 1
            if result.predicted_label == NO_TARGET:
                predicted_no_target[truth] += 1
            else:
                confusion[truth][result.predicted_label] += 1
        template_metrics = {
            "top1": sum(top1) / len(top1) if top1 else 0.0,
            "top3": sum(top3) / len(top3) if top3 else 0.0,
            "no_target_false_positive_rate": (
                sum(result.predicted_label != NO_TARGET for result in no_target_results)
                / len(no_target_results)
                if no_target_results
                else 0.0
            ),
            "latency_ms_p50": median(latencies) if latencies else 0.0,
            "latency_ms_p95": percentile(latencies, 0.95),
            "per_class_top1": {
                label: per_class_correct[label] / per_class_total[label]
                if per_class_total[label]
                else None
                for label in labels
            },
            "confusion_matrix_10": confusion,
            "predicted_no_target_by_class": dict(predicted_no_target),
        }
        metrics["templates"][template_name] = template_metrics
        gallery_lines.extend([f"## {template_name}", ""])
        for label in labels:
            examples = [
                (sample, result)
                for sample, result in target_results
                if sample.target_label == label
            ]
            correct = [
                pair for pair in examples if pair[1].predicted_label == label
            ][:2]
            failures = [
                pair for pair in examples if pair[1].predicted_label != label
            ][:2]
            gallery_lines.append(f"### {label}")
            for status, selected in (("correct", correct), ("fail", failures)):
                for sample, result in selected:
                    gallery_lines.append(
                        f"- {status}: `{sample.image_path}` → "
                        f"`{result.predicted_label}` ({result.confidence:.3f})"
                    )
            if not correct and not failures:
                gallery_lines.append("- no validation samples")
            gallery_lines.append("")
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (output_dir / "error_gallery.md").write_text(
        "\n".join(gallery_lines) + "\n", encoding="utf-8"
    )
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/dataset_v0.json")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "evaluation/experiments/exp_20260828_teacher_v0_prompts/artefacts"
        ),
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    repo_root = Path.cwd()
    manifest = DatasetManifest.read(args.manifest)
    backend = MobileClipTeacher(device=args.device)
    metrics = evaluate(repo_root, manifest, args.output_dir, backend)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
