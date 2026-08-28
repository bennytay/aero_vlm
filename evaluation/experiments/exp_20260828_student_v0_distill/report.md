# Experiment exp_20260828_student_v0_distill: compact embedding student

## Question

Can a MobileNetV2-0.35 student copy useful frozen MobileCLIP2-S0 embedding
rankings on the five supported targets while preserving the frozen reject rule?

## Hypothesis and exit gate

The student should beat the 10% random-label baseline on supported targets,
show non-random agreement with teacher decisions, and match static ONNX within
1e-5. No-target degradation is reported independently and is not hidden by
retuning T.

## Protocol

The 1,141 Phase 1 scenes were stratified by ground truth and deterministically
assigned by hashed source/scene ID: 913 train and 228 held out. No frames or
labels were added. The split contains 155 supported-target and 60 `no_target`
held-out scenes; parked counts are 4/1/3/3/2 and remain too thin for claims.

An uncentred 512-to-64 SVD basis was fitted only on train teacher image
embeddings. MobileNetV2 width 0.35 received 160x160 RGB and produced an L2-
normalised 64-D embedding. It trained from scratch for 30 CPU epochs using
cosine distillation plus 0.1-weighted in-batch contrastive loss. All teacher
embeddings were cached and detached; MobileCLIP2-S0 was never fine-tuned.

Teacher text embeddings were projected through the same train-only basis.
Teacher and student were compared on identical held-out scenes, ten target
texts, the default photo prompt, and frozen T=0.170. Parked classes are listed
separately in the local full metrics and are not part of the headline.

## Results

| Metric | Frozen teacher | Student |
|---|---:|---:|
| Supported top-1 | 78.71% | 40.00% |
| Supported macro recall | 79.56% | 35.70% |
| `no_target` FP | 23.33% | 100.00% |
| `no_target` TNR | 76.67% | 0.00% |

Supported per-class recall (teacher → student):

- water bottle: 66.67% → 30.30%
- black vehicle: 90.00% → 60.00%
- bicycle: 84.62% → 51.28%
- cardboard box: 90.91% → 18.18%
- sports ball: 65.63% → 18.75%

Parked recall is reported separately and is descriptive only:

| Parked class | n | Teacher | Student |
|---|---:|---:|---:|
| red backpack | 4 | 100.00% | 25.00% |
| person in high-vis | 1 | 100.00% | 0.00% |
| solar panel | 3 | 100.00% | 33.33% |
| wheelie bin | 3 | 66.67% | 0.00% |
| orange bucket | 2 | 100.00% | 0.00% |

Student/teacher thresholded decision agreement was 28.95%; mean cosine to the
projected teacher image target was 0.490. The student has 478,112 parameters,
a 2,274,059-byte checkpoint, and a 1,930,364-byte ONNX model. CPU model-only
latency was 8.89 ms p50 / 9.45 ms p95. Static ONNX matched PyTorch with maximum
absolute error 8.23e-7, passing the 1e-5 parity gate.

## Failures and limitations

The student beats the simple random-label baseline but is not mission-ready.
Most importantly, every held-out empty scene crosses the teacher's frozen
T=0.170 after the 64-D projection. Projection and student score scales are not
interchangeable with raw 512-D teacher cosines; a future training-only reject
objective or independently justified student calibration set is required.

Cardboard-box and sports-ball transfer are weak despite adequate teacher
recall. The from-scratch student and small dataset limit generalisation. MPS
was advertised by PyTorch but rejected by the active macOS runtime, so training
ran on CPU. Thin parked classes, unverified colour adjectives, and the lack of
aerial imagery remain blockers. Parked-class recall is not interpretable at
n=1–4.

## Conclusion

The representation/export part of the Phase 2 exit gate passes: supported
top-1 is 40%, four times the 10% random-label baseline, and ONNX parity is well
inside tolerance. The rejection part fails decisively. This establishes a
reproducible compact-student baseline and a blocker, not Phase 2 mission
success. Do not proceed to firmware or flight work from this result.

Local gitignored artefacts:

- `artefacts/student_v0.pt`
- `artefacts/student_v0.onnx`
- `artefacts/teacher_projection.npy`
- `artefacts/metrics.json`
- `artefacts/predictions.csv`
