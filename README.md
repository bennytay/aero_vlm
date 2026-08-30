# µAeroTrack

µAeroTrack is a low-cost onboard aerial object-detection and multi-object
tracking project. A companion computer processes the drone's live camera stream,
assigns boxes and stable track IDs, and records results while an independent
flight controller remains responsible for stabilisation, motors, pilot input,
and failsafes.

The first MVP is perception-only during piloted flight. It does not send movement
commands to the flight controller.

## Project documents

- [Project brief](docs/project_brief_v2.md)
- [Step-by-step implementation plan](docs/implementation_plan_v2.md)
- [First-principles article](docs/building_low_cost_aerial_tracking.md)

## Planned technical baseline

- VisDrone detection and tracking categories
- COCO-pretrained YOLO26n, with YOLO11n as a compatibility fallback
- ByteTrack baseline and a camera-motion-compensated tracker comparison
- Raspberry Pi 5 plus Hailo as the first flight target
- MaixCAM2 as the later cost-down target
- INT8 inference, bounded latest-frame capture, and selective small-object tiling

## Development setup

Python 3.11 and [`uv`](https://docs.astral.sh/uv/) are required:

```shell
uv sync
uv run pytest
uv run ruff check .
uv run mypy src tests
```

The default environment remains lightweight. Dataset, detector-training, export,
and accelerator dependencies will be optional.

## Current status

Phase 0 is complete. It provides:

- the frozen [`vocabulary_detection_v1.yaml`](configs/vocabulary_detection_v1.yaml);
- immutable detection, detection-frame, track, and track-frame contracts;
- checked-in JSON Schemas under `interfaces/schemas/`;
- strict box, label, timing, lifecycle, staleness, and duplicate-ID validation;
- a schema generator at `scripts/generate_schemas.py`; and
- experiment records with model format, input size, precision, pipeline boundary,
  dataset split, and duration.

Regenerate and verify contracts with:

```shell
uv run python scripts/generate_schemas.py
uv run pytest
```

Phase 1 is the public detector smoke test.
