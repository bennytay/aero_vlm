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

Phase 1 is implemented. It provides a pinned public YOLO26n smoke-test path,
strict conversion from supported COCO classes into the v1 detection contract,
image/folder/video inference, an annotated tracked video preview, JSONL and
latency artefacts, checked ONNX export, and IoU-based PyTorch/ONNX parity.

Install the optional runtime and run an image, folder, or video. The first run
downloads `yolo26n.pt` and refuses it unless its SHA-256 matches the checked-in
model config:

```shell
uv sync --group detection
uv run wam-detect run path/to/image-or-folder-or-video --output-dir runs/phase1
uv run wam-detect export --output-dir models
uv run wam-detect parity path/to/image.jpg --onnx models/yolo26n.onnx
```

The public checkpoint is COCO-trained. Phase 1 maps only unambiguous overlapping
classes: `person`, `bicycle`, `car`, `motorcycle` to `motor`, `bus`, and `truck`.
It cannot emit the VisDrone-only `pedestrian`, `van`, `tricycle`, or
`awning-tricycle` labels; those remain Phase 3 fine-tuning work.

Ultralytics code and released model artefacts are offered under
AGPL-3.0-or-later or an Ultralytics Enterprise licence. Confirm the appropriate
licence before distributing a product built on this baseline.

Phase 2 is partially complete. VisDrone-DET (train/val/test-dev) downloads
and converts end to end with real, verified results: 6471/548/1610 images,
343205/38759/75102 trainable boxes, disjoint-split validation, and a resize
report showing 45% of val boxes already fall below 8px minimum dimension at
640 input size (80% at 320) — see
[`exp_20260831_phase2_visdrone_det`](evaluation/experiments/exp_20260831_phase2_visdrone_det/report.md).
VisDrone-MOT/VID and UAVDT have no scriptable public mirror; their converters
are implemented and unit-tested against synthetic fixtures but not yet run
against real data, since manual downloads are required. See
[`data/README.md`](data/README.md) for both paths:

```shell
uv sync --group dataset
uv run wam-dataset visdrone-det-download --split all
uv run wam-dataset visdrone-det-convert --split all
uv run wam-dataset validate-splits \
  --manifest data/manifests/visdrone_det_train_v1.json \
  --manifest data/manifests/visdrone_det_val_v1.json \
  --manifest data/manifests/visdrone_det_test_v1.json
uv run wam-dataset report --manifest data/manifests/visdrone_det_val_v1.json
```
