"""Strict VLM contracts, teacher boundaries, and image-only inference."""

from wam_drones.vlm.contracts import (
    AnswerResponse,
    CaptionResponse,
    PointResponse,
    VLMInferenceRecord,
    VLMResponseContract,
    VLMSupervisionRecord,
    parse_vlm_response,
)
from wam_drones.vlm.inference import TransformersVLMBackend, VLMBackend, infer_image
from wam_drones.vlm.model_config import VLMModelConfig, load_model_config
from wam_drones.vlm.teacher import (
    FrameProvenance,
    TeacherArtifact,
    TeacherRunMetadata,
    load_teacher_artifact,
    validate_teacher_artifact,
    validate_track_jsonl,
)

__all__ = [
    "AnswerResponse",
    "CaptionResponse",
    "FrameProvenance",
    "PointResponse",
    "TeacherArtifact",
    "TeacherRunMetadata",
    "TransformersVLMBackend",
    "VLMBackend",
    "VLMInferenceRecord",
    "VLMModelConfig",
    "VLMResponseContract",
    "VLMSupervisionRecord",
    "infer_image",
    "load_model_config",
    "load_teacher_artifact",
    "parse_vlm_response",
    "validate_teacher_artifact",
    "validate_track_jsonl",
]
