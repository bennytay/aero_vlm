"""Public detection and tracking contracts."""

from wam_drones.detection.contracts import (
    CONTRACT_VERSION,
    TRACK_STALE_AFTER_FRAMES,
    Detection,
    FrameDetections,
    FrameTracks,
    TrackObservation,
    validate_frame_sequence,
)
from wam_drones.detection.inference import detect_image, detect_video
from wam_drones.detection.vocabulary import (
    DETECTION_ID_BY_LABEL,
    DETECTION_LABEL_BY_ID,
    DetectionLabel,
    DetectionVocabulary,
    load_detection_vocabulary,
)

__all__ = [
    "CONTRACT_VERSION",
    "DETECTION_ID_BY_LABEL",
    "DETECTION_LABEL_BY_ID",
    "TRACK_STALE_AFTER_FRAMES",
    "Detection",
    "DetectionLabel",
    "DetectionVocabulary",
    "FrameDetections",
    "FrameTracks",
    "TrackObservation",
    "detect_image",
    "detect_video",
    "load_detection_vocabulary",
    "validate_frame_sequence",
]
