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

Phase 2 is mostly complete. VisDrone-DET (train/val/test-dev, scriptable
download) and VisDrone-MOT (train/val, manual download) are both assembled
with real, verified results: 6471/548/1610 DET images and 24201/2846 MOT
frames, disjoint-split validation across all five manifests, and a resize
report showing 45% of DET val boxes already fall below 8px minimum dimension
at 640 input size (80% at 320). Cross-dataset dedup also found a genuine
upstream leak worth knowing about: 30 duplicate groups cross a DET/MOT
train-eval split boundary (e.g. 22 DET-val images are near-duplicates of 564
MOT-train frames) — see
[`exp_20260831_phase2_visdrone_det`](evaluation/experiments/exp_20260831_phase2_visdrone_det/report.md)
for the full breakdown and what it means for Phase 3/4 evaluation choices.
UAVDT has no scriptable public mirror and remains infrastructure-only
(converter implemented and unit-tested, no real data downloaded). See
[`data/README.md`](data/README.md) for all three paths:

```shell
uv sync --group dataset
uv run wam-dataset visdrone-det-download --split all
uv run wam-dataset visdrone-det-convert --split all
uv run wam-dataset visdrone-mot-convert --source-dir path/to/VisDrone2019-MOT-train --split train --source-url "..."
uv run wam-dataset validate-splits \
  --manifest data/manifests/visdrone_det_train_v1.json \
  --manifest data/manifests/visdrone_det_val_v1.json \
  --manifest data/manifests/visdrone_det_test_v1.json \
  --manifest data/manifests/visdrone_mot_train_v1.json \
  --manifest data/manifests/visdrone_mot_val_v1.json
uv run wam-dataset report --manifest data/manifests/visdrone_det_val_v1.json
```

Phase 3 is implemented but has not yet been run end-to-end. Its preparation
command keeps the raw official splits immutable while excluding the documented
DET-train/DET-val near-duplicate validation image from the primary score. It
uses 640px aspect-ratio-preserving letterboxing, deterministic training, native
VisDrone classes, and produces class-level AP-small/precision/recall/FP-per-
frame plus size, density, occlusion, and scene breakdowns. It also generates a
required 25 false-negative + 25 false-positive human review pack:

On the validated 8 GB RTX 3070 host, the Phase 3 configuration uses fixed
`batch: 8`. Ultralytics AutoBatch profiling terminated before selecting a batch;
a one-epoch fixed-batch probe completed cleanly at 8, while 16 triggered
assignment-step CUDA OOM retries. This is a hardware-specific training-policy
fallback, not an architecture, class, split, or augmentation change.

```shell
uv sync --group detection --group dataset
uv run wam-detect phase3-prepare \
  --output-dir evaluation/experiments/exp_20260831_phase3_visdrone_det/artefacts/dataset
uv run wam-detect phase3-evaluate \
  --checkpoint models/yolo26n.pt --model-kind coco \
  --manifest evaluation/experiments/exp_20260831_phase3_visdrone_det/artefacts/dataset/sequence_safe_val_manifest.json \
  --output-dir evaluation/experiments/exp_20260831_phase3_visdrone_det/artefacts/baseline
uv run wam-detect phase3-train \
  --data-yaml evaluation/experiments/exp_20260831_phase3_visdrone_det/artefacts/dataset/visdrone_det_phase3.yaml \
  --output-dir evaluation/experiments/exp_20260831_phase3_visdrone_det/artefacts
uv run wam-detect phase3-evaluate \
  --checkpoint evaluation/experiments/exp_20260831_phase3_visdrone_det/artefacts/train/weights/best.pt \
  --model-kind native \
  --manifest evaluation/experiments/exp_20260831_phase3_visdrone_det/artefacts/dataset/sequence_safe_val_manifest.json \
  --output-dir evaluation/experiments/exp_20260831_phase3_visdrone_det/artefacts/finetuned
```

The manual-review packs begin in a pending state and must be completed before
considering an architecture change.
