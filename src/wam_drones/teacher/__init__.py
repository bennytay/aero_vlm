"""Frozen MobileCLIP2-S0 teacher evaluation interfaces."""

from wam_drones.teacher.model import (
    MobileClipTeacher,
    TeacherResult,
    run_target,
)

__all__ = ["MobileClipTeacher", "TeacherResult", "run_target"]
