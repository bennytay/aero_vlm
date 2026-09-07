"""Offline VLM supervision helpers.

The VLM runtime is added in later phases.  This package currently contains
only the read-only tracking teacher boundary established in Step 0.
"""

from wam_drones.vlm.teacher import (
    FrameProvenance,
    TeacherArtifact,
    TeacherRunMetadata,
    load_teacher_artifact,
    validate_teacher_artifact,
    validate_track_jsonl,
)

__all__ = [
    "FrameProvenance",
    "TeacherArtifact",
    "TeacherRunMetadata",
    "load_teacher_artifact",
    "validate_teacher_artifact",
    "validate_track_jsonl",
]
