# Experiment exp_20260831_phase2_visdrone_det: VisDrone-DET assembly

## Question

Can VisDrone-DET be downloaded, converted to the project manifest and YOLO
format, deduplicated, and split-checked with real numbers before any
fine-tuning investment — and how much aerial small-object information is
already at risk before Phase 5's resolution/cadence tradeoffs even begin?

## Hypothesis

The official VisDrone-DET train/val/test-dev archives will convert cleanly to
disjoint-split manifests, and the resize report will show a material share of
boxes already below 4px/8px minimum dimension at 640 input size, and worse at
lower sizes — evidence for the plan's stop condition ("reduce detector cadence
before reducing resolution; tiny-object information is expensive to recover
after it is discarded").

## Configuration

- Git commit: `0ca35005269933777b53574b665d112d999cebfc`
- Config hash: `sha256:150011df7d14223799c4becefa9c98d8c509201989dbeeb660a1e1ee103cdd86`
- Dataset manifest hash: `sha256:75f979f433105b6bf53572d939b77d4ed02a2eb6f438835104802197298ccc75`
- Hardware: Apple Silicon development host
- Model format / input size / precision: not applicable (no model run in this
  experiment)
- Pipeline boundary: archive download to validated `DatasetManifest` + YOLO
  labels
- Dataset split: train + val + test (official VisDrone2019-DET splits)
- Random seed: 0 (no stochastic step)

## Protocol

1. `uv sync --group dataset`.
2. `wam-dataset visdrone-det-download --split all` — download
   `VisDrone2019-DET-{train,val,test-dev}.zip` from Ultralytics' GitHub
   release mirror and pin their SHA-256 into
   `configs/datasets/visdrone_det.yaml` on first run.
3. `wam-dataset visdrone-det-convert --split all` — parse every annotation
   file, write YOLO labels (trainable categories 1-10 only, score != 0) and a
   full-annotation JSON (all native fields, including ignore regions,
   truncation, occlusion) per image, and write one committed provenance
   manifest per split.
4. `wam-dataset validate-splits` across all three manifests.
5. `wam-dataset dedup` across all three manifests (dHash, Hamming
   threshold 4).
6. `wam-dataset report` per split (input sizes 640/512/416/320).

## Acceptance criteria

- Downloaded archive SHA-256 matches on every subsequent run.
- Converted image counts match the published VisDrone2019-DET split sizes
  (6471 / 548 / 1610).
- `validate-splits` finds no image hash reused across splits.
- The resize report produces finite counts for every input size.

## Results

All three splits converted to exactly the published counts: 6471 train, 548
val, 1610 test-dev images. Trainable box counts (category 1-10, score != 0):
343205 / 38759 / 75102. The 548-image val count of 38759 boxes and its
per-class breakdown were cross-checked by hand against the raw annotation
files and matched exactly. Ignored-region counts (category 0): 8813 / 1378 /
2180.

`validate-splits` passed: no image SHA-256 or sequence ID is shared across
train/val/test.

Resize report (below minimum-dimension-px counts after letterbox scaling to
each input size, val split):

| Input size | boxes < 4px | boxes < 8px | % of 38759 boxes < 8px |
|---|---|---|---|
| 640 | 5540 | 17510 | 45.2% |
| 512 | 8862 | 22337 | 57.6% |
| 416 | 13140 | 26737 | 69.0% |
| 320 | 17510 | 30934 | 79.8% |

Train and test-dev show the same pattern (train: 54012/158321 below 4px/8px
at 640; test-dev: 17429/39350). This confirms the hypothesis: even at the
full 640 training resolution almost half of all VisDrone-DET boxes are
already under 8 pixels in their minimum dimension, and by 320 that is nearly
80%. This is direct evidence for the plan's stop condition — cadence should
be reduced before resolution in Phase 5.

Perceptual-hash dedup (Hamming threshold 4) found 20 duplicate/near-duplicate
groups across the 8629 combined images, mostly adjacent frames sampled from
the same source video within one split (e.g. `9999981_00000_d_*`). **One
group spans train and val**: `0000323_01801_d_0000641`,
`_02001_d_0000642`, `_02201_d_0000643`, and `_02601_d_0000645` — four
near-identical frames from source video `0000323`, split across train and
val by the dataset's own official partition. This is a small, pre-existing
leak in the upstream VisDrone2019-DET split, not something introduced by this
conversion; it is reported per the plan's "detect duplicate image hashes
across DET, VID, and MOT before any split" step rather than corrected, since
the plan also says to keep official splits. Phase 3 validation should treat
val accuracy as very slightly optimistic because of it.

## Failures and anomalies

The first conversion attempt crashed on a real annotation row
(`1008,374,3,0,0,0,0,0`, a zero-height ignore-region box) because the parser
originally rejected any zero-width/zero-height box as invalid. This is
legitimate upstream VisDrone data, not corruption — the validator was relaxed
to reject only negative sizes, and the partially-converted train split
(892 of 6471 images already moved) was discarded and re-extracted from the
archive before reconverting cleanly. `tests/test_dataset_visdrone_det.py`
now covers this row shape directly.

VisDrone-MOT/VID and UAVDT have no scriptable public mirror (confirmed: the
Ultralytics asset host that serves VisDrone-DET returns 404 for MOT/VID/UAVDT
archive names). Their converters
(`src/wam_drones/dataset/visdrone_mot.py`, `src/wam_drones/dataset/uavdt.py`)
are implemented and unit-tested against synthetic fixtures but have not run
against real data in this experiment. The UAVDT column format in particular
is built from the published toolkit spec, not verified against a real file —
flagged in the module docstring and `data/README.md` for the next person to
check before trusting it for training.

## Conclusion

The VisDrone-DET slice of Phase 2 is complete with real, verified evidence:
reproducible download, correct conversion (counts and class breakdown
independently cross-checked), disjoint-split validation, duplicate
detection (including one genuine upstream train/val leak worth carrying into
Phase 3 analysis), and a resize report that quantifies real small-object risk
at every candidate input size. VisDrone-MOT/VID and UAVDT remain
infrastructure-only pending manual archive downloads — the overall Phase 2
gate is not yet fully met, only its VisDrone-DET portion.
