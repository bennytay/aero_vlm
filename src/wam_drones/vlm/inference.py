"""Image-only VLM inference and strict audit-record construction.

This module deliberately has no dependency on detection or tracking.  It accepts
one decoded RGB image at a time; callers that sample a video invoke it once per
selected frame.
"""

from __future__ import annotations

import hashlib
import json
import time
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol
from uuid import uuid4

from pydantic import ValidationError

from wam_drones.vlm.contracts import (
    PreprocessingMetadata,
    SourceFrame,
    VLMInferenceRecord,
    VLMResponseContract,
    parse_vlm_response,
)
from wam_drones.vlm.model_config import VLMModelConfig

if TYPE_CHECKING:
    from PIL.Image import Image


DecodeMode = Literal["unconstrained", "schema"]


class VLMBackend(Protocol):
    """Minimal image-only runtime boundary used by the CLI and smoke suite."""

    @property
    def model_revision(self) -> str: ...

    @property
    def preprocessing_dimensions(self) -> tuple[int, int]: ...

    def generate(self, rgb: Image, question: str) -> str: ...


def image_sha256(path: Path) -> str:
    """Return the source-file digest recorded in the inference audit."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TransformersVLMBackend:
    """Transformers implementation for chat-template image-text models.

    Imports are intentionally deferred so a normal detector-only installation
    remains lightweight.  The model is given a single PIL RGB image; no video
    frames, detector boxes, or tracks enter this boundary.
    """

    def __init__(
        self,
        config: VLMModelConfig,
        *,
        device: str = "auto",
        decode_mode: DecodeMode = "unconstrained",
    ) -> None:
        if decode_mode == "schema" and not config.supports_schema_constrained_decoding:
            raise ValueError(
                f"{config.name} does not support schema-constrained decoding"
            )
        try:
            transformers = import_module("transformers")
        except ImportError as error:
            raise RuntimeError(
                "run `uv sync --group vlm-inference` for VLM inference"
            ) from error
        self.config = config
        self.decode_mode = decode_mode
        self.processor = transformers.AutoProcessor.from_pretrained(
            config.repository, revision=config.revision
        )
        model_class = getattr(transformers, "AutoModelForMultimodalLM", None)
        if model_class is None:
            model_class = getattr(transformers, "AutoModelForImageTextToText", None)
        if model_class is None:
            raise RuntimeError(
                "installed Transformers lacks a multimodal auto-model loader"
            )
        model_options: dict[str, Any] = {"revision": config.revision}
        if device == "auto":
            model_options["device_map"] = "auto"
        self.model = model_class.from_pretrained(config.repository, **model_options)
        if device != "auto":
            self.model.to(device)
        self._model_revision = f"{config.repository}@{config.revision}"
        self._last_dimensions = (0, 0)

    @property
    def model_revision(self) -> str:
        return self._model_revision

    @property
    def preprocessing_dimensions(self) -> tuple[int, int]:
        return self._last_dimensions

    def _inputs(self, rgb: Image, question: str) -> Any:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": rgb,
                        "max_pixels": self.config.image_max_pixels,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[prompt], images=[rgb], return_tensors="pt")
        device = getattr(self.model, "device", None)
        return inputs.to(device) if device is not None else inputs

    def _schema_prefix_allowed_tokens(self) -> Any:
        """Build optional strict JSON token filtering for compatible models."""
        try:
            lmfe = import_module("lmformatenforcer")
            integration = import_module("lmformatenforcer.integrations.transformers")
        except ImportError as error:
            raise RuntimeError(
                "schema decoding requires `lm-format-enforcer`; install the "
                "vlm-inference group"
            ) from error
        parser = lmfe.JsonSchemaParser(VLMResponseContract.model_json_schema())
        return integration.build_transformers_prefix_allowed_tokens_fn(
            self.processor.tokenizer, parser
        )

    def generate(self, rgb: Image, question: str) -> str:
        if rgb.mode != "RGB":
            raise ValueError("VLMBackend requires an RGB image")
        inputs = self._inputs(rgb, question)
        pixel_values = inputs.get("pixel_values")
        if pixel_values is not None and hasattr(pixel_values, "shape"):
            shape = pixel_values.shape
            self._last_dimensions = (int(shape[-1]), int(shape[-2]))
        else:
            self._last_dimensions = rgb.size
        options: dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
        }
        if self.config.do_sample:
            options["temperature"] = self.config.temperature
        if self.decode_mode == "schema":
            options["prefix_allowed_tokens_fn"] = self._schema_prefix_allowed_tokens()
        output = self.model.generate(**inputs, **options)
        prompt_length = int(inputs["input_ids"].shape[-1])
        generated = output[:, prompt_length:]
        decoded = self.processor.batch_decode(generated, skip_special_tokens=True)
        return str(decoded[0]).strip()


def infer_image(
    image_path: Path,
    backend: VLMBackend,
    question: str,
    *,
    source_frame: SourceFrame | None = None,
    prompt_revision: str = "vlm-spike-v1",
) -> tuple[VLMInferenceRecord, float]:
    """Generate once, preserving malformed model output as a failed audit row."""
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "run `uv sync --group vlm-inference` for VLM inference"
        ) from error
    with Image.open(image_path) as opened:
        rgb = opened.convert("RGB")
    return infer_rgb(
        rgb,
        image_digest=image_sha256(image_path),
        backend=backend,
        question=question,
        source_frame=source_frame or SourceFrame(source_id=str(image_path)),
        prompt_revision=prompt_revision,
    )


def infer_rgb(
    rgb: Image,
    *,
    image_digest: str,
    backend: VLMBackend,
    question: str,
    source_frame: SourceFrame,
    prompt_revision: str = "vlm-spike-v1",
) -> tuple[VLMInferenceRecord, float]:
    """Generate from one already-decoded RGB frame and retain its source hash."""
    if rgb.mode != "RGB":
        raise ValueError("infer_rgb requires an RGB image")
    started = time.perf_counter()
    raw_generation = backend.generate(rgb, question)
    latency_ms = (time.perf_counter() - started) * 1000
    try:
        response = parse_vlm_response(raw_generation)
        parse_errors: tuple[str, ...] = ()
    except (ValidationError, ValueError, json.JSONDecodeError) as error:
        response = None
        parse_errors = (str(error),)
    model_width, model_height = backend.preprocessing_dimensions
    if model_width <= 0 or model_height <= 0:
        model_width, model_height = rgb.size
    return (
        VLMInferenceRecord(
            inference_id=str(uuid4()),
            image_sha256=image_digest,
            source_frame=source_frame,
            model_revision=backend.model_revision,
            prompt_revision=prompt_revision,
            preprocessing=PreprocessingMetadata(
                source_width_px=rgb.width,
                source_height_px=rgb.height,
                model_width_px=model_width,
                model_height_px=model_height,
            ),
            raw_generation=raw_generation,
            parse_errors=parse_errors,
            response=response,
        ),
        latency_ms,
    )
