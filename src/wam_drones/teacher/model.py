"""Frozen zero-shot MobileCLIP2-S0 adapter with lazy optional imports."""

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import monotonic_ns, perf_counter_ns
from typing import Any, Literal, Protocol, cast

from wam_drones.contracts import TargetObservation
from wam_drones.threshold import DEFAULT_SIMILARITY_THRESHOLD, apply_reject_threshold
from wam_drones.vocabulary import TARGET_IDS_BY_LABEL

MODEL_NAME = "MobileCLIP2-S0"
PRETRAINED = "dfndr2b"
NO_TARGET = "no_target"
PROMPT_TEMPLATES = {
    "photo": "a photo of a {label}",
    "aerial": "an aerial view of a {label}",
}
NO_TARGET_PROMPTS = {
    "photo": "a photo with none of these target objects",
    "aerial": "an aerial view with none of these target objects",
}


@dataclass(frozen=True)
class TeacherResult:
    """One zero-shot result plus evidence needed by aggregate evaluation."""

    predicted_label: str
    confidence: float
    probabilities: Mapping[str, float]
    similarities: Mapping[str, float]
    image_embedding: tuple[float, ...]
    latency_ms: float


class TeacherBackend(Protocol):
    """Small seam that keeps contract tests independent of torch."""

    def classify(self, image_path: Path, template_name: str) -> TeacherResult:
        """Classify an image against ten target prompts plus no-target."""
        ...


class MobileClipTeacher:
    """Frozen MobileCLIP2-S0 model used only under inference mode."""

    def __init__(
        self,
        device: str = "auto",
        *,
        decision_mode: Literal["threshold", "eleven_way"] = "threshold",
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> None:
        try:
            self._torch = importlib.import_module("torch")
            self._open_clip = importlib.import_module("open_clip")
            self._image_module = importlib.import_module("PIL.Image")
        except ImportError as error:
            raise RuntimeError(
                "teacher dependencies are optional; run `uv sync --extra training`"
            ) from error
        self.device = self._resolve_device(device)
        self.decision_mode = decision_mode
        self.threshold = threshold
        self._model, _, self._preprocess = self._open_clip.create_model_and_transforms(
            MODEL_NAME,
            pretrained=PRETRAINED,
            device=self.device,
        )
        self._model.eval()
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)
        self._tokenizer = self._open_clip.get_tokenizer(MODEL_NAME)
        self._text_features: dict[tuple[str, bool], Any] = {}

    def _resolve_device(self, requested: str) -> str:
        if requested != "auto":
            return requested
        if self._torch.cuda.is_available():
            return "cuda"
        if self._torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _synchronise(self) -> None:
        if self.device == "cuda":
            self._torch.cuda.synchronize()
        elif self.device == "mps":
            self._torch.mps.synchronize()

    def _prompts(
        self, template_name: str, *, include_no_target: bool
    ) -> tuple[str, ...]:
        try:
            template = PROMPT_TEMPLATES[template_name]
            no_target_prompt = NO_TARGET_PROMPTS[template_name]
        except KeyError as error:
            raise ValueError(f"unknown prompt template: {template_name}") from error
        prompts = tuple(template.format(label=label) for label in TARGET_IDS_BY_LABEL)
        return (*prompts, no_target_prompt) if include_no_target else prompts

    def _encoded_text(self, template_name: str, *, include_no_target: bool) -> Any:
        cache_key = (template_name, include_no_target)
        if cache_key not in self._text_features:
            tokens = self._tokenizer(
                self._prompts(template_name, include_no_target=include_no_target)
            ).to(self.device)
            with self._torch.inference_mode():
                features = self._model.encode_text(tokens)
                features /= features.norm(dim=-1, keepdim=True)
            self._text_features[cache_key] = features
        return self._text_features[cache_key]

    def text_embeddings(
        self, template_name: str, *, include_no_target: bool
    ) -> dict[str, tuple[float, ...]]:
        """Return normalised frozen text embeddings for cached-image analysis."""
        features = self._encoded_text(
            template_name, include_no_target=include_no_target
        )
        labels = tuple(TARGET_IDS_BY_LABEL)
        if include_no_target:
            labels = (*labels, NO_TARGET)
        rows = cast(list[list[float]], features.detach().cpu().tolist())
        return {label: tuple(row) for label, row in zip(labels, rows, strict=True)}

    def classify(self, image_path: Path, template_name: str) -> TeacherResult:
        include_no_target = self.decision_mode == "eleven_way"
        text_features = self._encoded_text(
            template_name, include_no_target=include_no_target
        )
        image = self._image_module.open(image_path).convert("RGB")
        image_tensor = self._preprocess(image).unsqueeze(0).to(self.device)
        self._synchronise()
        started = perf_counter_ns()
        with self._torch.inference_mode():
            image_features = self._model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            probabilities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        self._synchronise()
        latency_ms = (perf_counter_ns() - started) / 1_000_000
        labels = tuple(TARGET_IDS_BY_LABEL)
        if include_no_target:
            labels = (*labels, NO_TARGET)
        probability_values = cast(list[float], probabilities[0].detach().cpu().tolist())
        similarity_values = cast(
            list[float],
            (image_features @ text_features.T)[0].detach().cpu().tolist(),
        )
        by_label = dict(zip(labels, probability_values, strict=True))
        similarities = dict(zip(labels, similarity_values, strict=True))
        if self.decision_mode == "threshold":
            decision = apply_reject_threshold(similarities, self.threshold)
            predicted_label = decision.predicted_label or NO_TARGET
            confidence = min(1.0, max(0.0, decision.best_score))
        else:
            predicted_label = max(similarities, key=similarities.__getitem__)
            confidence = min(1.0, max(0.0, similarities[predicted_label]))
        embedding = cast(list[float], image_features[0].detach().cpu().tolist())
        return TeacherResult(
            predicted_label=predicted_label,
            confidence=confidence,
            probabilities=by_label,
            similarities=similarities,
            image_embedding=tuple(embedding),
            latency_ms=latency_ms,
        )


def run_target(
    image_path: Path,
    backend: TeacherBackend,
    *,
    template_name: str = "photo",
    timestamp_us: int | None = None,
) -> TargetObservation:
    """Return the shared observation contract using whole-image localisation."""
    result = backend.classify(image_path, template_name)
    timestamp = timestamp_us if timestamp_us is not None else monotonic_ns() // 1_000
    if result.predicted_label == NO_TARGET:
        return TargetObservation(
            timestamp_us=timestamp,
            visible=False,
            confidence=result.confidence,
            stale=False,
        )
    return TargetObservation(
        timestamp_us=timestamp,
        visible=True,
        confidence=result.confidence,
        center_x_normalized=0.5,
        center_y_normalized=0.5,
        bbox_xyxy_normalized=(0.0, 0.0, 1.0, 1.0),
        target_label=result.predicted_label,
        target_id=TARGET_IDS_BY_LABEL[result.predicted_label],
        stale=False,
    )
