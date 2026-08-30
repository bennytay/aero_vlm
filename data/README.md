# Data directory

Raw datasets must not be committed. Keep only versioned manifests, licences,
download instructions, hashes, and intentionally tiny test fixtures here.

Public aerial datasets are stored under ignored `data/raw/` paths. Manifests
retain source IDs, video sequence IDs, frame numbers, dimensions, annotation
versions, and hashes. Dataset splits must keep complete scenes and video
sequences together.

## Phase 1 smoke fixtures

Download an official VisDrone2019-DET train or validation archive from the
[VisDrone dataset repository](https://github.com/VisDrone/VisDrone-Dataset),
then extract a stable ten-image smoke set without committing the raw data:

```shell
uv run python scripts/prepare_phase1_fixtures.py path/to/VisDrone2019-DET-val.zip
uv run wam-detect run data/samples/phase1_visdrone --output-dir runs/phase1-images
```

## Phase 2 datasets

Install the dataset tooling once: `uv sync --group dataset`.

### VisDrone-DET (scriptable)

VisDrone-DET train/val/test-dev are mirrored on Ultralytics' GitHub release
assets, so they download and convert without any manual step:

```shell
uv run wam-dataset visdrone-det-download --split all
uv run wam-dataset visdrone-det-convert --split all
uv run wam-dataset validate-splits \
  --manifest data/manifests/visdrone_det_train_v1.json \
  --manifest data/manifests/visdrone_det_val_v1.json \
  --manifest data/manifests/visdrone_det_test_v1.json
uv run wam-dataset dedup \
  --manifest data/manifests/visdrone_det_train_v1.json \
  --manifest data/manifests/visdrone_det_val_v1.json \
  --manifest data/manifests/visdrone_det_test_v1.json
uv run wam-dataset report --manifest data/manifests/visdrone_det_val_v1.json
```

The first `visdrone-det-download` run pins each archive's SHA-256 into
`configs/datasets/visdrone_det.yaml`; later runs verify against it. Converted
images/YOLO labels/full annotations land under gitignored
`data/raw/visdrone_det/`; the committed provenance manifest
(`data/manifests/visdrone_det_<split>_v1.json`) holds no per-box geometry, so
it stays small enough to check in.

### VisDrone-MOT/VID (manual download, done for train+val)

No scriptable mirror exists (the host serving DET returns 404 for MOT/VID/
UAVDT filenames). Fetch `VisDrone2019-MOT-{train,val}.zip` from the Google
Drive/OneDrive/BaiduDisk links in the
[VisDrone dataset repository](https://github.com/VisDrone/VisDrone-Dataset)
— under "Task 4: Multi-Object Tracking" — prefer Google Drive over BaiduYun
(Baidu requires a Chinese phone/app for large files). Skip "Task 2: VID": it
ships the same underlying video sequences as MOT at identical archive sizes,
just packaged for a different challenge track; MOT's annotations are the
superset this project needs. Extract, then convert in place:

```shell
uv run wam-dataset visdrone-mot-convert \
  --source-dir path/to/VisDrone2019-MOT-train \
  --split train \
  --source-url "manual-download://visdrone-mot-train (Google Drive)"

uv run wam-dataset visdrone-mot-convert \
  --source-dir path/to/VisDrone2019-MOT-val \
  --split val \
  --source-url "manual-download://visdrone-mot-val (Google Drive)"
```

Real train+val numbers, cross-dataset dedup findings, and their implications
for Phase 3/4 evaluation choices are recorded in
`evaluation/experiments/exp_20260831_phase2_visdrone_det/report.md`.

### UAVDT (manual download required, not yet run)

Also no scriptable mirror; fetch from the links on its
[project page](https://sites.google.com/view/grli-uavdt). Extract, then:

```shell
uv run wam-dataset uavdt-convert \
  --source-dir path/to/UAVDT \
  --split train \
  --source-url "manual-download://uavdt-train (Google Drive)"
```

The UAVDT parser (`src/wam_drones/dataset/uavdt.py`) was built from the
published ground-truth format spec but has not been validated against real
UAVDT files, since none were available while writing it — check its output
on a couple of real sequences before trusting it for training; a column-count
mismatch raises instead of silently mis-parsing.
