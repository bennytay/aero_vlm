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

The design is complete. Phase 0 will freeze the detection vocabulary, contracts,
JSON Schemas, validation rules, and experiment metadata before model work begins.
