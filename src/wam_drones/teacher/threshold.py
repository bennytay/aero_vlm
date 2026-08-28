"""Build the Phase 1b rejection curve from saved teacher image embeddings."""

import argparse
import gzip
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from wam_drones.dataset import DatasetManifest
from wam_drones.teacher.model import MobileClipTeacher
from wam_drones.threshold import ScoredSample, choose_threshold, sweep_thresholds


def load_photo_embeddings(path: Path) -> dict[str, tuple[float, ...]]:
    """Load one saved photo-template embedding per source ID."""
    embeddings: dict[str, tuple[float, ...]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            row = cast(dict[str, Any], json.loads(line))
            if row["template"] == "photo":
                embeddings[str(row["source_id"])] = tuple(
                    float(value) for value in cast(list[float], row["embedding"])
                )
    if not embeddings:
        raise ValueError("no photo embeddings found")
    return embeddings


def cosine_scores(
    image_embedding: Sequence[float],
    text_embeddings: Mapping[str, Sequence[float]],
) -> dict[str, float]:
    """Dot already-normalised image and text embeddings deterministically."""
    return {
        label: sum(
            image_value * text_value
            for image_value, text_value in zip(
                image_embedding, text_embedding, strict=True
            )
        )
        for label, text_embedding in text_embeddings.items()
    }


def build_scored_samples(
    manifest: DatasetManifest,
    image_embeddings: Mapping[str, Sequence[float]],
    text_embeddings: Mapping[str, Sequence[float]],
) -> list[ScoredSample]:
    """Join the frozen manifest to cached embeddings by source ID."""
    samples: list[ScoredSample] = []
    for sample in manifest.samples:
        try:
            image_embedding = image_embeddings[sample.source_id]
        except KeyError as error:
            raise ValueError(f"missing image embedding: {sample.source_id}") from error
        samples.append(
            ScoredSample(
                expected_label=None if sample.no_target else sample.target_label,
                similarities=cosine_scores(image_embedding, text_embeddings),
            )
        )
    return samples


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/dataset_v0.json")
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path(
            "evaluation/experiments/exp_20260828_teacher_v0_prompts/"
            "artefacts/embeddings.jsonl.gz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "evaluation/experiments/exp_20260828_teacher_threshold/metrics.json"
        ),
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    manifest = DatasetManifest.read(args.manifest)
    image_embeddings = load_photo_embeddings(args.embeddings)
    teacher = MobileClipTeacher(device=args.device)
    text_embeddings = teacher.text_embeddings("photo", include_no_target=False)
    samples = build_scored_samples(manifest, image_embeddings, text_embeddings)
    thresholds = [round(0.10 + index * 0.005, 3) for index in range(101)]
    curve = sweep_thresholds(samples, thresholds)
    chosen = choose_threshold(curve)
    payload = {
        "experiment_id": "exp_20260828_teacher_threshold",
        "rule": "argmax ten target cosines, reject when best_score < threshold",
        "prompt_template": "a photo of a {label}",
        "current_11_way_baseline": {
            "overall_top1": 0.7598597721297108,
            "no_target_false_positive_rate": 0.48,
            "no_target_tnr": 0.52,
        },
        "chosen": chosen,
        "curve": curve,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(chosen, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
