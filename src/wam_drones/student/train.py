"""Train, export, and compare the Phase 2 embedding student."""

import argparse
import csv
import json
import platform
import random
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from time import perf_counter_ns
from typing import Any, cast

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms  # type: ignore[import-untyped]

from wam_drones.scope import PARKED_LABELS, SUPPORTED_LABELS
from wam_drones.student.data import (
    DistillationSample,
    deterministic_split,
    load_samples,
    project_normalized,
    projection_basis,
)
from wam_drones.student.model import StudentEmbeddingModel
from wam_drones.teacher.model import MobileClipTeacher
from wam_drones.threshold import apply_reject_threshold
from wam_drones.vocabulary import TARGET_IDS_BY_LABEL


class ImageEmbeddingDataset(Dataset[tuple[Tensor, Tensor, int]]):
    """Decode scenes lazily and return projected teacher targets."""

    def __init__(
        self, samples: list[DistillationSample], targets: np.ndarray[Any, Any]
    ) -> None:
        self.samples = samples
        self.targets = targets
        self.transform = transforms.Compose(
            [
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, int]:
        with Image.open(self.samples[index].image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, torch.from_numpy(self.targets[index]), index


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    macos_version = platform.mac_ver()[0]
    if (
        torch.backends.mps.is_available()
        and macos_version
        and tuple(map(int, macos_version.split(".")[:2])) >= (14, 0)
    ):
        return "mps"
    return "cpu"


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def classification_metrics(
    samples: list[DistillationSample],
    similarities: np.ndarray[Any, Any],
    threshold: float,
) -> dict[str, Any]:
    labels = tuple(TARGET_IDS_BY_LABEL)
    predictions = []
    for row in similarities:
        decision = apply_reject_threshold(
            dict(zip(labels, map(float, row), strict=True)), threshold
        )
        predictions.append(decision.predicted_label)
    per_class: dict[str, float] = {}
    for label in labels:
        indices = [
            i for i, sample in enumerate(samples) if sample.expected_label == label
        ]
        per_class[label] = (
            sum(predictions[i] == label for i in indices) / len(indices)
            if indices
            else 0.0
        )
    supported_indices = [
        i
        for i, sample in enumerate(samples)
        if sample.expected_label in SUPPORTED_LABELS
    ]
    no_target_indices = [
        i for i, sample in enumerate(samples) if sample.expected_label is None
    ]
    supported_top1 = sum(
        predictions[i] == samples[i].expected_label for i in supported_indices
    ) / len(supported_indices)
    no_target_fp = sum(predictions[i] is not None for i in no_target_indices) / len(
        no_target_indices
    )
    return {
        "supported_top1": supported_top1,
        "supported_macro_recall": sum(per_class[x] for x in SUPPORTED_LABELS)
        / len(SUPPORTED_LABELS),
        "supported_per_class_recall": {x: per_class[x] for x in SUPPORTED_LABELS},
        "parked_per_class_recall": {x: per_class[x] for x in PARKED_LABELS},
        "no_target_false_positive_rate": no_target_fp,
        "no_target_tnr": 1.0 - no_target_fp,
        "n_supported": len(supported_indices),
        "n_no_target": len(no_target_indices),
        "n_per_class": {
            label: sum(sample.expected_label == label for sample in samples)
            for label in labels
        },
        "predictions": predictions,
    }


def latency_ms(model: StudentEmbeddingModel, device: str) -> dict[str, float]:
    example = torch.zeros(1, 3, 160, 160, device=device)
    values = []
    model.eval()
    with torch.inference_mode():
        for _ in range(10):
            _ = model(example)
        for _ in range(100):
            started = perf_counter_ns()
            _ = model(example)
            if device == "mps":
                torch.mps.synchronize()
            elif device == "cuda":
                torch.cuda.synchronize()
            values.append((perf_counter_ns() - started) / 1_000_000)
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    seed = 20260828
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    samples = load_samples(
        root,
        root / "data/manifests/dataset_v0.json",
        root
        / "evaluation/experiments/exp_20260828_teacher_v0_prompts/artefacts"
        / "embeddings.jsonl.gz",
    )
    train_samples, eval_samples = deterministic_split(samples, seed, 0.2)
    basis = projection_basis(train_samples, 64)
    train_teacher = np.stack([sample.teacher_embedding for sample in train_samples])
    train_targets = project_normalized(train_teacher, basis)
    dataset = ImageEmbeddingDataset(train_samples, train_targets)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, generator=generator)
    device = resolve_device(args.device)
    model = StudentEmbeddingModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for images, targets, _ in loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            cosine_loss = (1.0 - (outputs * targets).sum(dim=-1)).mean()
            logits = outputs @ targets.T / 0.07
            contrastive_loss = F.cross_entropy(
                logits, torch.arange(len(images), device=device)
            )
            loss = cosine_loss + 0.1 * contrastive_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(images)
        history.append(total / len(dataset))
        print(f"epoch={epoch + 1} loss={history[-1]:.6f}", flush=True)

    output = root / "evaluation/experiments/exp_20260828_student_v0_distill/artefacts"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "student_v0.pt"
    torch.save(
        {"model": model.state_dict(), "projection": basis, "seed": seed}, checkpoint
    )
    np.save(output / "teacher_projection.npy", basis)

    eval_teacher = np.stack([sample.teacher_embedding for sample in eval_samples])
    eval_targets = project_normalized(eval_teacher, basis)
    eval_dataset = ImageEmbeddingDataset(eval_samples, eval_targets)
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
    model.eval()
    student_rows: list[np.ndarray[Any, Any]] = []
    with torch.inference_mode():
        for images, _, _ in eval_loader:
            student_rows.append(model(images.to(device)).cpu().numpy())
    student_embeddings = np.concatenate(student_rows)

    teacher = MobileClipTeacher(device=device)
    text_by_label = teacher.text_embeddings("photo", include_no_target=False)
    labels = tuple(TARGET_IDS_BY_LABEL)
    teacher_text = np.stack([text_by_label[label] for label in labels]).astype(
        np.float32
    )
    student_text = project_normalized(teacher_text, basis)
    teacher_similarities = eval_teacher @ teacher_text.T
    student_similarities = student_embeddings @ student_text.T
    teacher_metrics = classification_metrics(eval_samples, teacher_similarities, 0.17)
    student_metrics = classification_metrics(eval_samples, student_similarities, 0.17)
    agreement = sum(
        left == right
        for left, right in zip(
            teacher_metrics["predictions"], student_metrics["predictions"], strict=True
        )
    ) / len(eval_samples)
    teacher_metrics.pop("predictions")
    student_predictions = cast(list[str | None], student_metrics.pop("predictions"))

    onnx_path = output / "student_v0.onnx"
    model_cpu = model.cpu().eval()
    example = torch.randn(1, 3, 160, 160)
    torch.onnx.export(
        model_cpu,
        (example,),
        onnx_path,
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes=None,
        opset_version=17,
        dynamo=False,
    )
    import onnxruntime as ort  # type: ignore[import-untyped]

    pytorch_output = model_cpu(example).detach().numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_output = session.run(None, {"image": example.numpy()})[0]
    parity_max_abs = float(np.max(np.abs(pytorch_output - onnx_output)))

    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source_id", "expected_label", "student_prediction"])
        for sample, prediction in zip(eval_samples, student_predictions, strict=True):
            writer.writerow(
                [
                    sample.source_id,
                    sample.expected_label or "no_target",
                    prediction or "no_target",
                ]
            )
    metrics = {
        "seed": seed,
        "train_scenes": len(train_samples),
        "held_out_scenes": len(eval_samples),
        "threshold": 0.17,
        "teacher": teacher_metrics,
        "student": student_metrics,
        "student_teacher_decision_agreement": agreement,
        "mean_teacher_embedding_cosine": float(
            np.mean(np.sum(student_embeddings * eval_targets, axis=-1))
        ),
        "training_loss": history,
        "parameters": sum(parameter.numel() for parameter in model_cpu.parameters()),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint),
        "onnx_bytes": onnx_path.stat().st_size,
        "onnx_sha256": sha256_file(onnx_path),
        "onnx_parity_max_abs": parity_max_abs,
        "desktop_latency_ms": latency_ms(model_cpu, "cpu"),
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
