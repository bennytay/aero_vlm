# Experiment exp_20260831_phase2_visdrone_det: VisDrone-DET + VisDrone-MOT assembly

## Question

Can VisDrone-DET and VisDrone-MOT be downloaded, converted to the project
manifest and YOLO format, deduplicated, and split-checked with real numbers
before any fine-tuning or tracking investment — and does cross-dataset dedup
surface any accidental overlap between the detector-training data and the
tracking-evaluation data before it silently inflates measured accuracy?

## Hypothesis

The official archives will convert cleanly to disjoint-split manifests, the
resize report will show a material share of boxes already below 4px/8px at
640 input size and worse at lower sizes, and cross-dataset dedup will find
any accidental train/eval overlap between DET and MOT before it contaminates
Phase 3/4 results.

## Configuration

- Git commit: `25a016562dedcf6772278668593e98fdbd0a718a`
- Config hash: `sha256:150011df7d14223799c4becefa9c98d8c509201989dbeeb660a1e1ee103cdd86`
- Dataset manifest hash: `sha256:e3ab9de5125e8b7436639260bbd4cf5968e0ae4add06ae55f3967feed83507f6`
- Hardware: Apple Silicon development host
- Model format / input size / precision: not applicable (no model run)
- Pipeline boundary: archive download/extraction to validated
  `DatasetManifest` + YOLO labels
- Dataset split: VisDrone-DET train+val+test and VisDrone-MOT train+val
  (official splits)
- Random seed: 0 (no stochastic step)

## Protocol

1. `uv sync --group dataset`.
2. VisDrone-DET: `wam-dataset visdrone-det-download` +
   `visdrone-det-convert --split all` (Ultralytics' GitHub mirror, scriptable).
3. VisDrone-MOT: manually downloaded `VisDrone2019-MOT-{train,val}.zip` from
   the official Google Drive links (no scriptable mirror exists — confirmed
   the host serving DET returns 404 for MOT/VID/UAVDT filenames), extracted
   locally, then `wam-dataset visdrone-mot-convert --split {train,val}
   --source-dir <extracted dir>`.
4. `wam-dataset validate-splits` across all 5 resulting manifests.
5. `wam-dataset dedup` across all 5 manifests (dHash, Hamming threshold 4).
6. `wam-dataset report` per manifest (input sizes 640/512/416/320).

## Acceptance criteria

- Converted image/frame counts match the archives' real contents.
- `validate-splits` finds no image hash or sequence ID reused across splits.
- `dedup` runs across all datasets together, not per-dataset in isolation.
- The resize report produces finite counts for every input size.

## Results

### VisDrone-DET

All three splits converted to exactly the published counts: 6471 train, 548
val, 1610 test-dev images; 343205 / 38759 / 75102 trainable boxes; 8813 /
1378 / 2180 ignored regions. The val count (38759 boxes, full per-class
breakdown) was cross-checked by hand against the raw annotation files and
matched exactly.

At val split: 45.2% of boxes are already below 8px minimum dimension at 640
input size, rising to 79.8% at 320. Train and test-dev show the same pattern.
This is direct evidence for the plan's stop condition — reduce detector
cadence before resolution.

### VisDrone-MOT

Manually downloaded (`VisDrone2019-MOT-train.zip`, 56 sequences;
`VisDrone2019-MOT-val.zip`, 7 sequences) and converted. Frame counts match
the raw sequence directories exactly: 24201 train frames, 2846 val frames.
Trainable box counts: 1,105,516 / 114,132. Ignored regions: 62369 / 3994. At
640 input size, val already has 21742/114132 (19%) boxes below 8px minimum
dimension.

A first conversion attempt crashed with a `relative_to` path error: the
MOT/UAVDT converters read frame images directly from the user's external
source directory (e.g. `~/Downloads/VisDrone2019-MOT-train`) without ever
copying them into the project's own `data/raw/` tree, unlike the DET
converter (which moves images from its own throwaway extraction directory).
This only worked in unit tests because the test fixtures happened to nest
the fake source directory under the same temp path used as `repo_root`. Both
converters were fixed to copy each frame into `data/raw/{visdrone_mot,uavdt}/
<split>/<sequence>/images/` before hashing and recording its path (mirroring
the DET converter, but using copy rather than move since the source is the
user's own downloaded archive, not ours to consume). The regression tests
were also fixed to place their fixtures outside `repo_root`, so this class
of bug can't hide in test fixtures again.

### Split validation and dedup

`validate-splits` passed across all 5 manifests: no image SHA-256 or
sequence ID is shared across any split.

Perceptual-hash dedup (Hamming threshold 4) across all 35,676 combined
images/frames found 2984 duplicate/near-duplicate groups — the large jump
from DET-alone's 20 groups is expected, since MOT frames are sampled at
video frame rate and adjacent frames are near-identical by design (most
groups are consecutive frames within one sequence).

**The important result: 30 of those groups cross a train/eval split
boundary**, entirely between VisDrone-DET and VisDrone-MOT (VisDrone's own
authors built both from overlapping raw footage):

| Crossing | Groups | DET images touched | MOT frames touched |
|---|---|---|---|
| DET-train ↔ MOT-val | 8 | 11 | 134 |
| DET-val ↔ MOT-train | 21 | 22 | 564 |
| DET-train ↔ DET-val ↔ MOT-train | 1 | 33 (incl. the Phase 1 finding) | — |

`DET-test` (test-dev) is untouched by any split-crossing group (0 matches) —
it remains a clean held-out set with respect to both DET-train and any MOT
split.

This is real, upstream overlap in VisDrone's own video/image collection, not
something this conversion introduced, and per the plan ("keep official
splits") it is reported rather than corrected. Its practical implication for
this project:

- If VisDrone-MOT-train frames are ever folded into detector training data
  (the MOT converter also emits YOLO labels), do not evaluate that detector
  on VisDrone-DET-val — 22 DET-val images are near-duplicates of 564
  MOT-train frames, which would silently inflate val accuracy.
- Symmetrically, a detector trained on DET-train should not be assumed
  "unseen" by MOT-val when judging tracking-pipeline detection quality — 11
  DET-train images are near-duplicates of 134 MOT-val frames.
- DET-test (test-dev) is the safest choice for a truly held-out detector
  accuracy check against this specific leak.

## Failures and anomalies

1. The VisDrone-DET parser initially crashed on a real annotation row
   (`1008,374,3,0,0,0,0,0`, a zero-height ignore-region box) because it
   rejected any zero-size box as invalid. Relaxed to reject only negative
   sizes; covered by a regression test. The partially-converted train split
   (892/6471 images already moved) was discarded and re-extracted before
   reconverting cleanly.
2. The MOT/UAVDT converters' missing copy-into-repo step (see above) crashed
   on the first real MOT-train run. Fixed in both converters and covered by
   regression tests that place fixtures outside `repo_root`.

## Conclusion

VisDrone-DET and VisDrone-MOT are both now assembled with real, verified
data: reproducible download (DET) or documented manual download (MOT),
correct conversion (counts cross-checked by hand), disjoint-split validation,
and a resize report quantifying real small-object risk at every candidate
input size. Cross-dataset dedup found a genuine, actionable train/eval
overlap between DET and MOT that should shape Phase 3/4 evaluation choices.
UAVDT remains infrastructure-only (implemented and unit-tested, no real data
downloaded) — the Phase 2 gate is now met for VisDrone-DET and VisDrone-MOT;
UAVDT is the one remaining gap.
