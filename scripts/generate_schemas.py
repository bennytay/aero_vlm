"""Generate the checked-in JSON Schemas for public runtime contracts."""

import json
from pathlib import Path

from pydantic import BaseModel

from wam_drones.dataset.manifest import DatasetManifest, ImageRecord
from wam_drones.detection import (
    Detection,
    FrameDetections,
    FrameTracks,
    TrackObservation,
)
from wam_drones.vlm.contracts import (
    VLMInferenceRecord,
    VLMResponseContract,
    VLMSupervisionRecord,
)

SCHEMAS: dict[str, type[BaseModel]] = {
    "detection_v1.schema.json": Detection,
    "frame_detections_v1.schema.json": FrameDetections,
    "frame_tracks_v1.schema.json": FrameTracks,
    "track_observation_v1.schema.json": TrackObservation,
    "image_record_v1.schema.json": ImageRecord,
    "dataset_manifest_v1.schema.json": DatasetManifest,
    "vlm_response_v1.schema.json": VLMResponseContract,
    "vlm_supervision_record_v1.schema.json": VLMSupervisionRecord,
    "vlm_inference_record_v1.schema.json": VLMInferenceRecord,
}


def main() -> int:
    """Write stable, human-readable schemas under interfaces/schemas."""
    repo_root = Path(__file__).resolve().parents[1]
    schema_dir = repo_root / "interfaces" / "schemas"
    schema_dir.mkdir(parents=True, exist_ok=True)
    for filename, model in SCHEMAS.items():
        path = schema_dir / filename
        path.write_text(
            json.dumps(model.model_json_schema(), indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
