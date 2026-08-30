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
