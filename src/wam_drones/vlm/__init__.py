"""Offline VLM supervision helpers.

The VLM runtime is added in later phases.  This package currently contains
only the read-only tracking teacher boundary established in Step 0.
"""

from wam_drones.vlm.contracts import (
    AnswerResponse,
    CaptionResponse,
    PointResponse,
    VLMInferenceRecord,
    VLMResponseContract,
    VLMSupervisionRecord,
    parse_vlm_response,
)
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
    "VLMInferenceRecord",
    "VLMResponseContract",
    "VLMSupervisionRecord",
    "load_teacher_artifact",
    "parse_vlm_response",
    "validate_teacher_artifact",
    "validate_track_jsonl",
]
