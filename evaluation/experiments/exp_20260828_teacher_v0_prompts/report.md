# Experiment exp_20260828_teacher_v0_prompts: Frozen teacher prompt baseline

## Question

Does frozen MobileCLIP2-S0 beat chance on the held-out v0 validation set, and
which of the two frozen prompt templates works better?

## Hypothesis

Both prompt templates will exceed the 1-in-11 chance baseline without model
training or fine-tuning.

## Configuration

- Model: MobileCLIP2-S0, OpenCLIP pretrained tag `dfndr2b`
- Model mode: frozen, `eval()`, inference mode, no gradients
- Model hash: `sha256:ab91a1a0c4330d6b1913e24d5035dfdea15423316aaec649610c6b1c6ddd0e95`
- Dataset manifest: `data/manifests/dataset_v0.json`
- Dataset hash: `sha256:a42b737ae10557d906af9d161df6e273ee8d315e9511f7034bb8bbdeac35947d`
- Git commit at experiment start: `e4a3073e07fbc2ea5fa773e9463aa451062706a6`
- Hardware: MacBook Air, Apple M2, 16 GB RAM, MPS
- Software: Python 3.11, torch 2.13.0, OpenCLIP 3.3.0, Pillow 12.3.0
- Firmware: N/A
- Random seed: N/A; validation scene IDs are sorted deterministically

The Apple checkpoint is used under the Apple ML Research Model Terms of Use.
Public image licence and attribution URLs are retained per manifest sample.

## Dataset

The manifest contains 1,141 unique validation scenes. Open Images records use
their per-image CC BY licence from official metadata. The Construction Site
Safety records are CC BY 4.0. Splits are inherited from official validation
splits; adjacent video frames are not randomly divided.

| Ground truth | n |
|---|---:|
| red backpack | 20 |
| water bottle | 167 |
| person in high-vis | 2 |
| black vehicle | 200 |
| solar panel | 14 |
| wheelie bin | 17 |
| bicycle | 195 |
| cardboard box | 57 |
| sports ball | 158 |
| orange bucket | 11 |
| no_target | 300 |

The small classes remain in the evaluation and are not padded with invented
labels. Open Images does not verify the adjectives red, black, orange, or
cardboard. `Bottle`, `Waste container`, and `Ball` are also broader than the
project labels. These are known label-noise limitations, not verified colour
ground truth.

## Protocol

For each image and each prompt template, embed the image and eleven texts: the
ten exact frozen labels plus one no-target prompt. Rank cosine similarities
after normalisation and softmax scaling. Measure image inference latency after
model and text features are loaded.

Prompt templates:

1. `a photo of a {label}`
2. `an aerial view of a {label}`

No-target prompts use the matching prefix followed by “with none of these
target objects”. Whole-image localisation is returned for Phase 1 because
MobileCLIP classification does not produce a box.

## Acceptance criteria

- Overall top-1 exceeds 1/11 chance (9.09%).
- Most target classes exceed 10% class-wise top-1.
- Counts, failures, no-target false positives, and latency are reported.

## Results

| Metric | Photo | Aerial |
|---|---:|---:|
| Top-1 | **75.99%** | **73.88%** |
| Top-3 | **95.97%** | **92.81%** |
| no_target false-positive rate | 48.00% | 67.67% |
| Latency p50 | 14.95 ms | 14.59 ms |
| Latency p95 | 19.02 ms | 18.03 ms |

Both templates beat chance by a large margin. The photo template wins overall
top-1 and top-3 and rejects no-target scenes materially better. Every target
class is above the 10% class-wise chance reference, although the thin classes
do not support strong conclusions.

## Failures by class

| Class | Photo top-1 | Aerial top-1 | Main observed failures |
|---|---:|---:|---|
| red backpack (n=20) | 90.0% | 90.0% | Two images predicted person in high-vis; colour is unverified. |
| water bottle (n=167) | 71.9% | 86.2% | Photo often predicts no_target (31) or orange bucket (8). |
| person in high-vis (n=2) | 100% | 100% | Count is far too small to generalise. |
| black vehicle (n=200) | 96.5% | 95.5% | A few no_target and cross-class predictions; black is unverified. |
| solar panel (n=14) | 92.9% | 92.9% | One image predicted person in high-vis; class is thin. |
| wheelie bin (n=17) | 94.1% | 94.1% | One image predicted cardboard box; source class is broader. |
| bicycle (n=195) | 93.8% | 94.9% | Photo predicts no_target for seven scenes. |
| cardboard box (n=57) | 82.5% | 78.9% | Confused with water bottle/person; cardboard is unverified. |
| sports ball (n=158) | 71.5% | 78.5% | Often predicts no_target (38 photo, 25 aerial). |
| orange bucket (n=11) | 54.5% | 72.7% | Confused with person/wheelie bin/no_target; orange is unverified. |
| no_target (n=300) | 52.0% TNR | 32.3% TNR | The no-target prompt is poorly calibrated. |

## Artefacts

Generated artefacts are retained locally under the ignored `artefacts/`
directory and can be reproduced from the committed manifest.

| Artefact | SHA-256 |
|---|---|
| `metrics.json` | `7145d5a5af00e72a8cd0197117c988f31ee32855cf0a3fe44b5b7d780bf1534b` |
| `predictions.csv` | `a571d0c6c21d3236b51ac8beec7135b62cf8c85baf70e9b7effa605cda4b0995` |
| `embeddings.jsonl.gz` | `caaa9f05f28a39e465fcba680a708ff57330e5d2d10751d209c803b03aadb88d` |
| `error_gallery.md` | `cf7b533cddaf717ea7d9ce0341371a16a18b947298c5603a236448a725484690` |

## Conclusion

The frozen teacher passes the Phase 1 exit gate: it is materially above chance
overall and for most classes. The photo template is the better v0 default.
No-target calibration and the six thin/noisy classes are the main weaknesses.

The next step is Phase 2 compact-student distillation, gated on preserving
these dataset caveats and reporting results separately for thin classes. No
Phase 2 implementation is included here.
