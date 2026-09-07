"""Versioned, strict contracts for VLM responses and offline supervision."""

from __future__ import annotations

import re
from typing import Annotated, Final, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    TypeAdapter,
    model_validator,
)

VLM_RESPONSE_VERSION: Final = "vlm_response_v1"
VLM_SUPERVISION_RECORD_VERSION: Final = "vlm_supervision_record_v1"
VLM_INFERENCE_RECORD_VERSION: Final = "vlm_inference_record_v1"
COORDINATE_SYSTEM: Final = "relative_0_1000_xy"


class FrozenContract(BaseModel):
    """Immutable contract base that rejects unknown fields and coercion."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class CaptionResponse(FrozenContract):
    schema_version: Literal["vlm_response_v1"] = VLM_RESPONSE_VERSION
    type: Literal["caption"]
    status: Literal["ok", "unknown"]
    text: str | None = None

    @model_validator(mode="after")
    def validate_status(self) -> CaptionResponse:
        if self.status == "ok" and not self.text:
            raise ValueError("caption text is required when status is ok")
        if self.status != "ok" and self.text is not None:
            raise ValueError("caption text is forbidden when status is not ok")
        return self


class BooleanAnswer(FrozenContract):
    kind: Literal["boolean"]
    value: bool


class CountAnswer(FrozenContract):
    kind: Literal["count"]
    value: int = Field(ge=0)


class LabelAnswer(FrozenContract):
    kind: Literal["label"]
    value: str = Field(min_length=1)


class TextAnswer(FrozenContract):
    kind: Literal["text"]
    value: str = Field(min_length=1)


AnswerValue: TypeAlias = Annotated[
    BooleanAnswer | CountAnswer | LabelAnswer | TextAnswer,
    Field(discriminator="kind"),
]


class AnswerResponse(FrozenContract):
    schema_version: Literal["vlm_response_v1"] = VLM_RESPONSE_VERSION
    type: Literal["answer"]
    status: Literal["ok", "not_visible", "ambiguous", "unknown"]
    answer: AnswerValue | None = None

    @model_validator(mode="after")
    def validate_status(self) -> AnswerResponse:
        if self.status == "ok" and self.answer is None:
            raise ValueError("answer is required when status is ok")
        if self.status != "ok" and self.answer is not None:
            raise ValueError("answer is forbidden when status is not ok")
        return self


class RelativePoint(FrozenContract):
    x: int = Field(ge=0, le=1000)
    y: int = Field(ge=0, le=1000)
    coordinate_system: Literal["relative_0_1000_xy"] = COORDINATE_SYSTEM
    semantics: Literal["target_center", "review_region"]


class PointResponse(FrozenContract):
    schema_version: Literal["vlm_response_v1"] = VLM_RESPONSE_VERSION
    type: Literal["point"]
    status: Literal["found", "not_visible", "ambiguous", "unknown", "no_candidate"]
    description: str | None = None
    point: RelativePoint | None = None

    @model_validator(mode="after")
    def validate_status(self) -> PointResponse:
        if self.status == "found" and self.point is None:
            raise ValueError("point is required when status is found")
        if self.status != "found" and self.point is not None:
            raise ValueError("point is forbidden when status is not found")
        return self


VLMResponse: TypeAlias = Annotated[
    CaptionResponse | AnswerResponse | PointResponse,
    Field(discriminator="type"),
]
_RESPONSE_ADAPTER: TypeAdapter[VLMResponse] = TypeAdapter(VLMResponse)


class VLMResponseContract(RootModel[VLMResponse]):
    """Schema-generation wrapper for the discriminated VLM response union."""


_CODE_FENCE_PATTERN: Final = re.compile(
    r"^```(?:json)?\s*\n?(?P<body>.*?)\n?```$", re.DOTALL
)


def _strip_code_fence(payload: str) -> str:
    """Unwrap a single leading/trailing Markdown code fence, if present."""
    stripped = payload.strip()
    match = _CODE_FENCE_PATTERN.match(stripped)
    return match.group("body") if match else stripped


def parse_vlm_response(payload: str | bytes | bytearray) -> VLMResponse:
    """Validate a raw model generation.

    The only normalization performed is unwrapping a single Markdown code
    fence some models emit despite being told to reply with bare JSON; no
    other repair is attempted.
    """
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8")
    return _RESPONSE_ADAPTER.validate_json(_strip_code_fence(payload))


def pixel_to_relative_0_1000_xy(
    x_px: int, y_px: int, *, image_width_px: int, image_height_px: int
) -> tuple[int, int]:
    """Convert an in-bounds pixel coordinate to the contract's inclusive grid."""
    if image_width_px < 2 or image_height_px < 2:
        raise ValueError("image dimensions must be at least two pixels")
    if not (0 <= x_px < image_width_px and 0 <= y_px < image_height_px):
        raise ValueError("pixel coordinate is outside the source image")
    return (
        round(x_px * 1000 / (image_width_px - 1)),
        round(y_px * 1000 / (image_height_px - 1)),
    )


def relative_0_1000_xy_to_pixel(
    x: int, y: int, *, image_width_px: int, image_height_px: int
) -> tuple[int, int]:
    """Convert a validated inclusive response-grid coordinate to source pixels."""
    RelativePoint(x=x, y=y, semantics="target_center")
    if image_width_px < 2 or image_height_px < 2:
        raise ValueError("image dimensions must be at least two pixels")
    return (
        round(x * (image_width_px - 1) / 1000),
        round(y * (image_height_px - 1) / 1000),
    )


class SourceFrame(FrozenContract):
    """Source identity retained in audit and supervision records only."""

    source_id: str = Field(min_length=1)
    sequence_id: str | None = None
    frame_number: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_sequence_frame(self) -> SourceFrame:
        if (self.sequence_id is None) != (self.frame_number is None):
            raise ValueError("sequence_id and frame_number must be supplied together")
        return self


class PreprocessingMetadata(FrozenContract):
    source_width_px: int = Field(gt=0)
    source_height_px: int = Field(gt=0)
    model_width_px: int = Field(gt=0)
    model_height_px: int = Field(gt=0)


class VLMInferenceRecord(FrozenContract):
    """Immutable audit envelope; failed parses remain visible as failures."""

    schema_version: Literal["vlm_inference_record_v1"] = VLM_INFERENCE_RECORD_VERSION
    inference_id: str = Field(min_length=1)
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_frame: SourceFrame
    model_revision: str = Field(min_length=1)
    prompt_revision: str = Field(min_length=1)
    preprocessing: PreprocessingMetadata
    raw_generation: str
    parse_errors: tuple[str, ...] = ()
    response: VLMResponse | None = None

    @model_validator(mode="after")
    def validate_parse_outcome(self) -> VLMInferenceRecord:
        if self.response is None and not self.parse_errors:
            raise ValueError("parse errors are required when response is absent")
        if self.response is not None and self.parse_errors:
            raise ValueError("parse errors are forbidden when response is present")
        return self


class VLMSupervisionRecord(FrozenContract):
    """Canonical, model-agnostic training evidence before chat export."""

    schema_version: Literal["vlm_supervision_record_v1"] = (
        VLM_SUPERVISION_RECORD_VERSION
    )
    sample_id: str = Field(min_length=1)
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_frame: SourceFrame
    dataset: str = Field(min_length=1)
    source_split: str = Field(min_length=1)
    split: Literal["train", "development", "test"]
    task_family: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected_response: VLMResponse
    prompt_template_id: str = Field(min_length=1)
    teacher_source: Literal["ground_truth", "tracks", "detector", "human"]
    teacher_evidence: dict[str, object]
    label_confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    rejection_reasons: tuple[str, ...] = ()
